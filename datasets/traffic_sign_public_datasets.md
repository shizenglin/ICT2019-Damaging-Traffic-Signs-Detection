# Public datasets

## Dashcam videos/images

- [BelgiumTS](http://btsd.ethz.ch/shareddata/index.html)
    - 13480 annotated signs in 9006 images
    - 210 classes in 11 superclasses:
        - triangles (corresponds to Danger superclass in GTSDB)
        - redcircles (corresponds to Prohibitory superclass in GTSDB)
        - bluecircles (corresponds to Mandatory superclass in GTSDB)
        - redbluecircles
        - diamonds
        - revtriangle
        - stop
        - forbidden
        - squares
        - rectanglesup
        - rectanglesdown
- [DITS (Italian)](http://www.dis.uniroma1.it/~bloisi/ds/dits.html)
    - 478 annotated signs in 459 images
    - 3 classes (no further subclasses):
        - Prohibitory (circles)
        - Warning (triangle)
        - Indication (square)
        - NOTE: With respect to previous dataset as the German Traffic Sign Recognition Benchmark, the superclasses are different. Particularly there is the 'indication' superclass that is completely new while the "german prohibitory" and "german mandatory" are here merged to form the prohibitory superclass.
- [GTSDB (german)](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)
    - 1213 annotated signs in 900 images
    - 3 superclasses:
        - Prohibitive
        - Danger
        - Mandatory
- [STSD (Swedish)](http://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/)
    - set 1: 1484 annotated signs in 1970 images
    - set 2: 1356 annotated signs in 1807 images
    - 13 classes in 4 superclasses:
        - Prohibitory
        - Warning
        - Mandatory
        - Information
- [LISA (USA)](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html)
    - 7855 annotated signs in 6610 images
- [Tsinghua-Tencent 100K (Chinese)](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
    - 30000 annotated signs in 10000 images
    - 45 classes of >100 examples in 3 superclass:
        - Warning
        - Prohibitory
        - Mandatory
    - Also contains pixel masks

## Cropped images (sign only)

- [GTSRB (German)](http://benchmark.ini.rub.de/?section=gtsrb)
    - 40+ classes with 50,000+ images
    - ClassIDs the same as GTSDB dataset
- [DITS (Italian)](http://www.dis.uniroma1.it/~bloisi/ds/dits.html)
    - 8648 images
    - 58 classes grouped in 3 superclasses:
        - Prohibitory (circles)
        - Warning (triangle)
        - Indication (square)
