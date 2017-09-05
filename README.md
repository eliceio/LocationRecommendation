# LocationRecommendation

__ELICE__, 2017 Summer Team Project

This project is supported by __Daejeon Center for Creative Economy & Innovation__

## Web Demo:
__http://1.255.55.102__

## Team members:

* Sumin Lim 
  * KAIST, Business and Technology Management
  * Algorithm Implementation

* Hyunji Lee 
  * JBNU, Division of Computer Science and Engineering
  * Algorithm Implementation
  * Web Development

* Minkyung Kim
  * KAIST, Electrical Engineering
  * Algorithm Implementation

* Junghyun Kim
  * Hanyang University, Department of Information System
  * Web Development

* Hyungtaek Choi 
  * JBNU, Division of Computer Science and Engineering



T.A.: Vinnam Kim / Jooyeon Kim 

## Requirements

Initial requirements are as follows

* Python 3.6.0
* Tensorflow 1.1.0
* Pandas 0.20.1
* Numpy 1.13.1
* Scipy 0.19.1
* BeautifulSoup 4.5.3
* urllib 3.6


## Usage

```
python3 main.py
```

* Parameter (You can modify these parameters in the main.py file)
  * Minimum number of logs, default=5
  * Maximum number of logs, default=5
  * Beta, default=10
  * Number of topic, default=8
  * Number of maxiter, default=300

* Output
  * MRR
  * Precision@N graph (Forward and backward recommendation)
  * Numpy file of recommended restaurants

## Reference
Takeshi Kurashima, Tomoharu Iwata, Takahide Hoshide, Noriko Takaya, Ko Fujimura (2013) __Geo topic model: joint modeling of user's activity area and interests for location recommendation__, *Proceedings of the sixth ACM international conference on Web search and data mining*  [doi>10.1145/2433396.2433444] 

