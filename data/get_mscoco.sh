mkdir -p mscoco
cd mscoco

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip train2014.zip
unzip annotations_trainval2014.zip
rm train2014.zip
rm annotations_trainval2014.zip
