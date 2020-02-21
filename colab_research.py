import numpy as np
import cv2
from samplelib import *

def main():
    batch_size = 16
    resolution = 128

    f = SampleProcessor.TypeFlags
    face_type = f.FACE_TYPE_FULL

    output_sample_types = [ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution] ]
    output_sample_types += [ [f.TRANSFORMED | face_type | f.MODE_BGR, resolution // (2**i) ] for i in range(ms_count)]
    output_sample_types += [ [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution // (2**i) ] for i in range(ms_count)]

    SampleGeneratorFace(self.training_data_src_path, batch_size=batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=True, scale_range=np.array([-0.05, 0.05]) ),
                        output_sample_types=output_sample_types )

if __name__ == "__main__":
    main()