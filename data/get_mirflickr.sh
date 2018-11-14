mkdir -p mirflickr
cd mirflickr

aria2c -x 16 -s 16 http://press.liacs.nl/mirflickr/mirflickr1m.v2/tags.zip
unzip tags.zip

for i in {0..9}
do
    aria2c -x 16 -s 16 http://press.liacs.nl/mirflickr/mirflickr1m.v2/images${i}.zip
    unzip -o images${i}.zip
done
