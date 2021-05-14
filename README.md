| import  |
| ------- |
| python3 |
| cv2     |
| dlib    |
| imutils |

## Step1 : landmarks detection

[Facial landmarks with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

Step #1: Localize the face in the image

* openCV의 Haar cascades를 사용하여 얼굴 검출

Step #2: Detect the key facial structures

* 입, 눈썹, 코, 턱을 필수적으로 localize, lavbel 해야한다
* dlib에 포함된 facial landmark detector를 이용



#### rect_to_bb 함수

dlib detector는 bounding box의 (x,y)좌표를 반환

openCV에선 bounding box의 입력을 (x,y,width,height)로 받음

때문에 rect_to_bb 함수가 (x,y)를 받아 (x,y,width,height)를 변환해줌

#### shape_to_np 함수

dlib face landmark detector가 68개 (x,y) 좌표의 객체를 반환

shpae_to_np는 이를 NumPy array로 변환해줌



```
python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 
```

```--shape-predictor``` : 학습된 landmark detector의 경로

You can download the detector model [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) or you can use the ***“Downloads”\*** section of this post to grab the code + example images + pre-trained detector as well.

```--image``` : 대상 이미지



## Step2 : detect_face_parts

[Detect eyes, nose, lips, and jaw with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/)

ROI(Region Of Interest) : 여기선 검출한 얼굴 또는 face part

shape predictor는 68개의 점을 반환 하는데 각각은 인덱스를 갖는다.  

* The mouth can be accessed through points [48, 68].
* The right eyebrow through points [17, 22].
* The left eyebrow through points [22, 27].
* The right eye using [36, 42].
* The left eye with [42, 48].
* The nose using [27, 35].
* And the jaw via [0, 17].

face_utils of the imutils library

```python
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])
```

visualize_facial_landmarks

```python
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()
	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
```



