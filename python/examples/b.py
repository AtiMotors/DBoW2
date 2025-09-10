import pydbow2
import cv2
import numpy as np

N =4
img_dir = "/home/arindam/mules/DBoW2/demo/images"
def load_features():
    orb = cv2.ORB_create()
    features = []
    for i in range(N):
        img = cv2.imread(img_dir + "/image" + str(i) + ".png")
        _, des = orb.detectAndCompute(img, None)
        features.append(des)
    return features

def test_vocabulary(features):
    k = 9
    L = 3
    weight = pydbow2.TF_IDF
    scoring = pydbow2.L1_NORM

    voc = pydbow2.Vocabulary(k = k, L = L, weight_type= weight, scoring = scoring)
    print("Creating a small ", k, "^", L , " vocabulary")
    voc.create(features, k, L)
    print("Done ")

    print("Vocabulary Info: \n", voc)

    print("Matching images against themselves (0 low, 1 high): ")

    for i in range(N):
        #v1 = voc.transform(features[i])
        v1, fv1 = voc.transform_get_features(features[i], 0)
        print(fv1)
        for j in range(N):
            v2 = voc.transform(features[j])
            score = voc.score(v1, v2)
            print("Image ", i, "vs Image ", j , ": ", score)

    

def test_database(features):
    print("Creating a small database  ")
    print("Loading Vocabulary...")
    voc = pydbow2.Vocabulary(path="/home/arindam/mules/DBoW2/build/small_voc.yml.gz")
    print("Vocabulary loaded. \n", voc)
    db = pydbow2.Database(voc)

    for i in range(N):
        db.add(features[i])

    print("Done creating database")
    print("Db size ", db.size())
    print("Db info: \n", db)

    print("Querying the db..")

    for i in range(N):
        result = db.query(features[i], 4);
        print("Searching for Image ", i , ". ", result )



def main():
    features = load_features()
    print("feature shape: ", len(features))
    for f in features:
        print("f shape ", f.shape)
    test_vocabulary(features)
    test_database(features)

if __name__=="__main__":
    main()
