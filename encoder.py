from os import listdir
from os.path import isfile, join
import skipthoughts
import os.path
import pickle

annotation_dir = 'coco/test_annotations'
encoded_vector_dir = 'coco/test_encoded'
garbage = 'coco/test_garbage'

model = skipthoughts.load_model()
onlyfiles = [f_ for f_ in listdir(annotation_dir) if isfile(join(annotation_dir, f_))]

onlyfiles_1 = onlyfiles[0:len(onlyfiles)//5]
onlyfiles_2 = onlyfiles[len(onlyfiles)//5:2*len(onlyfiles)//5]
onlyfiles_3 = onlyfiles[2*len(onlyfiles)//5:3*len(onlyfiles)//5]
onlyfiles_4 = onlyfiles[3*len(onlyfiles)//5:4*len(onlyfiles)//5]
onlyfiles_5 = onlyfiles[4*len(onlyfiles)//5:5*len(onlyfiles)//5]





for files in onlyfiles:
    if os.path.exists(join(encoded_vector_dir,files)) == False:
        with open(join(annotation_dir,files)) as f:
            captions = f.read().split(',')

        captions = [cap for cap in captions if len(cap.strip()) > 0]

        try:
            caption_vectors = skipthoughts.encode(model, captions)
        except:
            with open(join(garbage,files), mode='w') as myfile:
                myfile.write(' ')
            continue
        files_ = files[0:-4]+'.pkl'

        with open(join(encoded_vector_dir,files_), mode='wb') as myfile:
            pickle.dump(caption_vectors, myfile)
    else:
        print("skipped")
