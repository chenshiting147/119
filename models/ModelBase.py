import os
import json
import time
import inspect
import pickle
import colorsys
import imagelib
from pathlib import Path
from utils import Path_utils
from utils import std_utils
from utils.cv2_utils import *
import numpy as np
import cv2
from samplelib import SampleGeneratorBase
from nnlib import nnlib
from interact import interact as io

'''
You can implement your own model. Check examples.
'''


class ModelBase(object):
    def __init__(self, model_path, training_data_src_path=None, training_data_dst_path=None, pretraining_data_path=None,
                 debug=False, device_args=None,
                 ask_write_preview_history=True,
                 ask_target_iter=True,
                 ask_batch_size=True,
                 ask_sort_by_yaw=True,
                 ask_random_flip=True,
                 ask_src_scale_mod=True):

        device_args['force_gpu_idx'] = device_args.get('force_gpu_idx', -1)
        device_args['cpu_only'] = device_args.get('cpu_only', False)

        if device_args['force_gpu_idx'] == -1 and not device_args['cpu_only']:
            idxs_names_list = nnlib.device.getValidDevicesIdxsWithNamesList()
            if len(idxs_names_list) > 1:
                io.log_info("系统中有多个gpu: ")
                for idx, name in idxs_names_list:
                    io.log_info("[%d] : %s" % (idx, name))

                device_args['force_gpu_idx'] = io.input_int("选择哪个GPU idx? ( skip: 最好 GPU ) : ", -1,
                                                            [x[0] for x in idxs_names_list])
        self.device_args = device_args

        self.device_config = nnlib.DeviceConfig(allow_growth=False, **self.device_args)

        io.log_info("加载模型...")

        self.model_path = model_path
        self.model_data_path = Path(self.get_strpath_storage_for_file('data.dat'))

        self.training_data_src_path = training_data_src_path
        self.training_data_dst_path = training_data_dst_path
        self.pretraining_data_path = pretraining_data_path

        self.src_images_paths = None
        self.dst_images_paths = None
        self.src_yaw_images_paths = None
        self.dst_yaw_images_paths = None
        self.src_data_generator = None
        self.dst_data_generator = None
        self.debug = debug
        self.is_training_mode = (training_data_src_path is not None and training_data_dst_path is not None)

        self.iter = 0
        self.options = {}
        self.loss_history = []
        self.sample_for_preview = None

        model_data = {}
        if self.model_data_path.exists():
            model_data = pickle.loads(self.model_data_path.read_bytes())
            self.iter = max(model_data.get('iter', 0), model_data.get('epoch', 0))
            if 'epoch' in self.options:
                self.options.pop('epoch')
            if self.iter != 0:
                self.options = model_data['options']
                self.loss_history = model_data['loss_history'] if 'loss_history' in model_data.keys() else []
                self.sample_for_preview = model_data[
                    'sample_for_preview'] if 'sample_for_preview' in model_data.keys() else None

        ask_override = self.is_training_mode and self.iter != 0 and io.input_in_time("按回车键2秒后覆盖模型设置.",
                                                                                     5 if io.is_colab() else 2)

        yn_str = {True: 'y', False: 'n'}

        if self.iter == 0:
            io.log_info("\n第一次运行模型.为每次运行输入默认的模型选项.")

        if ask_write_preview_history and (self.iter == 0 or ask_override):
            default_write_preview_history = False if self.iter == 0 else self.options.get('write_preview_history',
                                                                                          False)
            self.options['write_preview_history'] = io.input_bool(
                "写入预览历史? (y/n ?:帮助 默认:%s) : " % (yn_str[default_write_preview_history]), default_write_preview_history,
                help_message="预览历史记录将写入modelname_history文件夹.")
        else:
            self.options['write_preview_history'] = self.options.get('write_preview_history', False)

        if (self.iter == 0 or ask_override) and self.options['write_preview_history'] and io.is_support_windows():
            choose_preview_history = io.input_bool("选择图像作为预览历史? (y/n 默认:%s) : " % (yn_str[False]), False)
        else:
            choose_preview_history = False

        if ask_target_iter:
            if (self.iter == 0 or ask_override):
                self.options['target_iter'] = max(0, io.input_int("目标迭代 (默认:unlimited/default) : ", 0))
            else:
                self.options['target_iter'] = max(model_data.get('target_iter', 0), self.options.get('target_epoch', 0))
                if 'target_epoch' in self.options:
                    self.options.pop('target_epoch')

        if ask_batch_size and (self.iter == 0 or ask_override):
            default_batch_size = 0 if self.iter == 0 else self.options.get('batch_size', 0)
            self.options['batch_size'] = max(0, io.input_int("批量大小 (?:帮助 默认:%d) : " % (default_batch_size),
                                                             default_batch_size,
                                                             help_message="批量越大的神经网络泛化效果越好,但会导致内存不足.手动调整视频卡的这个值."))
        else:
            self.options['batch_size'] = self.options.get('batch_size', 0)

        if ask_sort_by_yaw:
            if (self.iter == 0):
                self.options['sort_by_yaw'] = io.input_bool("是否按照脸部角度进行排序训练? (y/n ?:帮助 默认:n) : ", False,
                                                            help_message="神经网络不会学习与dst人脸方向不匹配的src人脸方向.如果dst面部有覆盖下颌的毛发,请不要使用.")
            else:
                self.options['sort_by_yaw'] = self.options.get('sort_by_yaw', False)

        if ask_random_flip:
            if (self.iter == 0):
                self.options['random_flip'] = io.input_bool("随机翻转脸部? (y/n ?:帮助 默认:y) : ", True,
                                                            help_message="如果没有这个选项,预测人脸看起来会更自然,但是src faceset应该像dst faceset一样覆盖所有的人脸方向.")
            else:
                self.options['random_flip'] = self.options.get('random_flip', True)

        if ask_src_scale_mod:
            if (self.iter == 0):
                self.options['src_scale_mod'] = np.clip(io.input_int("Src 脸部标度修正 % ( -30...30, ?:帮助 默认:0) : ", 0,
                                                                     help_message="如果src脸型比dst宽,试着降低这个值以得到更好的结果."), -30,
                                                        30)
            else:
                self.options['src_scale_mod'] = self.options.get('src_scale_mod', 0)

        self.write_preview_history = self.options.get('write_preview_history', False)
        if not self.write_preview_history and 'write_preview_history' in self.options:
            self.options.pop('write_preview_history')

        self.target_iter = self.options.get('target_iter', 0)
        if self.target_iter == 0 and 'target_iter' in self.options:
            self.options.pop('target_iter')

        self.batch_size = self.options.get('batch_size', 0)
        self.sort_by_yaw = self.options.get('sort_by_yaw', False)
        self.random_flip = self.options.get('random_flip', True)

        self.src_scale_mod = self.options.get('src_scale_mod', 0)
        if self.src_scale_mod == 0 and 'src_scale_mod' in self.options:
            self.options.pop('src_scale_mod')

        self.onInitializeOptions(self.iter == 0, ask_override)

        nnlib.import_all(self.device_config)
        self.keras = nnlib.keras
        self.K = nnlib.keras.backend

        self.onInitialize()

        self.options['batch_size'] = self.batch_size

        if self.debug or self.batch_size == 0:
            self.batch_size = 1

        if self.is_training_mode:
            if self.device_args['force_gpu_idx'] == -1:
                self.preview_history_path = self.model_path / ('%s_history' % (self.get_model_name()))
            else:
                self.preview_history_path = self.model_path / (
                '%d_%s_history' % (self.device_args['force_gpu_idx'], self.get_model_name()))

            if self.write_preview_history or io.is_colab():
                if not self.preview_history_path.exists():
                    self.preview_history_path.mkdir(exist_ok=True)
                else:
                    if self.iter == 0:
                        for filename in Path_utils.get_image_paths(self.preview_history_path):
                            Path(filename).unlink()

            if self.generator_list is None:
                raise ValueError('您没有设置训练数据生成器')
            else:
                for i, generator in enumerate(self.generator_list):
                    if not isinstance(generator, SampleGeneratorBase):
                        raise ValueError('训练数据生成器不是SampleGeneratorBase的子类')

            if (self.sample_for_preview is None) or (self.iter == 0):

                if self.iter == 0:
                    if choose_preview_history and io.is_support_windows():
                        wnd_name = "[p] - next. [enter] - confirm."
                        io.named_window(wnd_name)
                        io.capture_keys(wnd_name)
                        choosed = False
                        while not choosed:
                            self.sample_for_preview = self.generate_next_sample()
                            preview = self.get_static_preview()
                            io.show_image(wnd_name, (preview * 255).astype(np.uint8))

                            while True:
                                key_events = io.get_key_events(wnd_name)
                                key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(
                                    key_events) > 0 else (0, 0, False, False, False)
                                if key == ord('\n') or key == ord('\r'):
                                    choosed = True
                                    break
                                elif key == ord('p'):
                                    break

                                try:
                                    io.process_messages(0.1)
                                except KeyboardInterrupt:
                                    choosed = True

                        io.destroy_window(wnd_name)
                    else:
                        self.sample_for_preview = self.generate_next_sample()

        model_summary_text = []

        model_summary_text += ["===== 模型概况 ====="]
        model_summary_text += ["== 模型名称: " + self.get_model_name()]
        model_summary_text += ["=="]
        model_summary_text += ["== 当前迭代: " + str(self.iter)]
        model_summary_text += ["=="]
        model_summary_text += ["== 模型选项:"]
        for key in self.options.keys():
            model_summary_text += ["== |== %s : %s" % (key, self.options[key])]

        if self.device_config.multi_gpu:
            model_summary_text += ["== |== 多 gpu : True "]

        model_summary_text += ["== 运行在:"]
        if self.device_config.cpu_only:
            model_summary_text += ["== |== [CPU]"]
        else:
            for idx in self.device_config.gpu_idxs:
                model_summary_text += ["== |== [%d : %s]" % (idx, nnlib.device.getDeviceName(idx))]

        if not self.device_config.cpu_only and self.device_config.gpu_vram_gb[0] == 2:
            model_summary_text += ["=="]
            model_summary_text += ["== 警告:您正在使用2GB GPU.结果质量明显下降."]
            model_summary_text += ["== 如果训练没有开始,关闭所有程序,然后重试."]
            model_summary_text += ["== 你也可以禁用Windows Aero桌面来获得额外的免费VRAM."]
            model_summary_text += ["=="]

        model_summary_text += ["========================="]
        model_summary_text = "\r\n".join(model_summary_text)
        self.model_summary_text = model_summary_text
        io.log_info(model_summary_text)

    # overridable
    def onInitializeOptions(self, is_first_run, ask_override):
        pass

    # overridable
    def onInitialize(self):
        '''
        initialize your keras models

        store and retrieve your model options in self.options['']

        check example
        '''
        pass

    # overridable
    def onSave(self):
        # save your keras models here
        pass

    # overridable
    def onTrainOneIter(self, sample, generator_list):
        # train your keras models here

        # return array of losses
        return (('loss_src', 0), ('loss_dst', 0))

    # overridable
    def onGetPreview(self, sample):
        # you can return multiple previews
        # return [ ('preview_name',preview_rgb), ... ]
        return []

    # overridable if you want model name differs from folder name
    def get_model_name(self):
        return Path(inspect.getmodule(self).__file__).parent.name.rsplit("_", 1)[1]

    # overridable
    def get_converter(self):
        raise NotImplementeError
        # return existing or your own converter which derived from base

    def get_target_iter(self):
        return self.target_iter

    def is_reached_iter_goal(self):
        return self.target_iter != 0 and self.iter >= self.target_iter

    # multi gpu in keras actually is fake and doesn't work for training https://github.com/keras-team/keras/issues/11976
    # def to_multi_gpu_model_if_possible (self, models_list):
    #    if len(self.device_config.gpu_idxs) > 1:
    #        #make batch_size to divide on GPU count without remainder
    #        self.batch_size = int( self.batch_size / len(self.device_config.gpu_idxs) )
    #        if self.batch_size == 0:
    #            self.batch_size = 1
    #        self.batch_size *= len(self.device_config.gpu_idxs)
    #
    #        result = []
    #        for model in models_list:
    #            for i in range( len(model.output_names) ):
    #                model.output_names = 'output_%d' % (i)
    #            result += [ nnlib.keras.utils.multi_gpu_model( model, self.device_config.gpu_idxs ) ]
    #
    #        return result
    #    else:
    #        return models_list

    def get_previews(self):
        return self.onGetPreview(self.last_sample)

    def get_static_preview(self):
        return self.onGetPreview(self.sample_for_preview)[0][1]  # first preview, and bgr

    def save(self):
        Path(self.get_strpath_storage_for_file('summary.txt')).write_text(self.model_summary_text)
        self.onSave()

        model_data = {
            'iter': self.iter,
            'options': self.options,
            'loss_history': self.loss_history,
            'sample_for_preview': self.sample_for_preview
        }
        self.model_data_path.write_bytes(pickle.dumps(model_data))

    def load_weights_safe(self, model_filename_list, optimizer_filename_list=[]):
        for model, filename in model_filename_list:
            filename = self.get_strpath_storage_for_file(filename)
            if Path(filename).exists():
                model.load_weights(filename)

        if len(optimizer_filename_list) != 0:
            opt_filename = self.get_strpath_storage_for_file('opt.h5')
            if Path(opt_filename).exists():
                try:
                    with open(opt_filename, "rb") as f:
                        d = pickle.loads(f.read())

                    for x in optimizer_filename_list:
                        opt, filename = x
                        if filename in d:
                            weights = d[filename].get('weights', None)
                            if weights:
                                opt.set_weights(weights)
                                print("set ok")
                except Exception as e:
                    print("Unable to load ", opt_filename)

    def save_weights_safe(self, model_filename_list, optimizer_filename_list=[]):
        for model, filename in model_filename_list:
            filename = self.get_strpath_storage_for_file(filename)
            model.save_weights(filename + '.tmp')

        rename_list = model_filename_list
        if len(optimizer_filename_list) != 0:
            opt_filename = self.get_strpath_storage_for_file('opt.h5')

            try:
                d = {}
                for opt, filename in optimizer_filename_list:
                    fd = {}
                    symbolic_weights = getattr(opt, 'weights')
                    if symbolic_weights:
                        fd['weights'] = self.K.batch_get_value(symbolic_weights)

                    d[filename] = fd

                with open(opt_filename + '.tmp', 'wb') as f:
                    f.write(pickle.dumps(d))

                rename_list += [('', 'opt.h5')]
            except Exception as e:
                print("Unable to save ", opt_filename)

        for _, filename in rename_list:
            filename = self.get_strpath_storage_for_file(filename)
            source_filename = Path(filename + '.tmp')
            if source_filename.exists():
                target_filename = Path(filename)
                if target_filename.exists():
                    target_filename.unlink()
                source_filename.rename(str(target_filename))

    def debug_one_iter(self):
        images = []
        for generator in self.generator_list:
            for i, batch in enumerate(next(generator)):
                if len(batch.shape) == 4:
                    images.append(batch[0])

        return imagelib.equalize_and_stack_square(images)

    def generate_next_sample(self):
        return [next(generator) for generator in self.generator_list]

    def train_one_iter(self):
        sample = self.generate_next_sample()
        iter_time = time.time()
        losses = self.onTrainOneIter(sample, self.generator_list)
        iter_time = time.time() - iter_time
        self.last_sample = sample

        self.loss_history.append([float(loss[1]) for loss in losses])

        if self.iter % 10 == 0:
            plist = []

            if io.is_colab():
                previews = self.get_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    plist += [(bgr, self.get_strpath_storage_for_file('preview_%s.jpg' % (name)))]

            if self.write_preview_history:
                plist += [(self.get_static_preview(), str(self.preview_history_path / ('%.6d.jpg' % (self.iter))))]

            for preview, filepath in plist:
                preview_lh = ModelBase.get_loss_history_preview(self.loss_history, self.iter, preview.shape[1],
                                                                preview.shape[2])
                img = (np.concatenate([preview_lh, preview], axis=0) * 255).astype(np.uint8)
                cv2_imwrite(filepath, img)

        self.iter += 1

        return self.iter, iter_time

    def pass_one_iter(self):
        self.last_sample = self.generate_next_sample()

    def finalize(self):
        nnlib.finalize_all()

    def is_first_run(self):
        return self.iter == 0

    def is_debug(self):
        return self.debug

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def get_iter(self):
        return self.iter

    def get_loss_history(self):
        return self.loss_history

    def set_training_data_generators(self, generator_list):
        self.generator_list = generator_list

    def get_training_data_generators(self):
        return self.generator_list

    def get_model_root_path(self):
        return self.model_path

    def get_strpath_storage_for_file(self, filename):
        if self.device_args['force_gpu_idx'] == -1:
            return str(self.model_path / (self.get_model_name() + '_' + filename))
        else:
            return str(self.model_path / (
            str(self.device_args['force_gpu_idx']) + '_' + self.get_model_name() + '_' + filename))

    def set_vram_batch_requirements(self, d):
        # example d = {2:2,3:4,4:8,5:16,6:32,7:32,8:32,9:48}
        keys = [x for x in d.keys()]

        if self.device_config.cpu_only:
            if self.batch_size == 0:
                self.batch_size = 2
        else:
            if self.batch_size == 0:
                for x in keys:
                    if self.device_config.gpu_vram_gb[0] <= x:
                        self.batch_size = d[x]
                        break

                if self.batch_size == 0:
                    self.batch_size = d[keys[-1]]

    @staticmethod
    def get_loss_history_preview(loss_history, iter, w, c):
        loss_history = np.array(loss_history.copy())

        lh_height = 100
        lh_img = np.ones((lh_height, w, c)) * 0.1
        loss_count = len(loss_history[0])
        lh_len = len(loss_history)

        l_per_col = lh_len / w
        plist_max = [[max(0.0, loss_history[int(col * l_per_col)][p],
                          *[loss_history[i_ab][p]
                            for i_ab in range(int(col * l_per_col), int((col + 1) * l_per_col))
                            ]
                          )
                      for p in range(loss_count)
                      ]
                     for col in range(w)
                     ]

        plist_min = [[min(plist_max[col][p], loss_history[int(col * l_per_col)][p],
                          *[loss_history[i_ab][p]
                            for i_ab in range(int(col * l_per_col), int((col + 1) * l_per_col))
                            ]
                          )
                      for p in range(loss_count)
                      ]
                     for col in range(w)
                     ]

        plist_abs_max = np.mean(loss_history[len(loss_history) // 5:]) * 2

        for col in range(0, w):
            for p in range(0, loss_count):
                point_color = [1.0] * c
                point_color[0:3] = colorsys.hsv_to_rgb(p * (1.0 / loss_count), 1.0, 1.0)

                ph_max = int((plist_max[col][p] / plist_abs_max) * (lh_height - 1))
                ph_max = np.clip(ph_max, 0, lh_height - 1)

                ph_min = int((plist_min[col][p] / plist_abs_max) * (lh_height - 1))
                ph_min = np.clip(ph_min, 0, lh_height - 1)

                for ph in range(ph_min, ph_max + 1):
                    lh_img[(lh_height - ph - 1), col] = point_color

        lh_lines = 5
        lh_line_height = (lh_height - 1) / lh_lines
        for i in range(0, lh_lines + 1):
            lh_img[int(i * lh_line_height), :] = (0.8,) * c

        last_line_t = int((lh_lines - 1) * lh_line_height)
        last_line_b = int(lh_lines * lh_line_height)

        lh_text = 'Iter: %d' % (iter) if iter != 0 else ''

        lh_img[last_line_t:last_line_b, 0:w] += imagelib.get_text_image((last_line_b - last_line_t, w, c), lh_text,
                                                                        color=[0.8] * c)
        return lh_img
