cd images
mkdir english-train english-test hindi-train hindi-test math-train math-test

cd english-train
cat ../ud_english_train | xargs -I{} -P 20 wget -q {}
ls | grep -v jpeg | xargs -I{} mv {} {}.jpeg


cd ../english-test
cat ../ud_english_test | xargs -I{} -P 20 wget -q {}
ls | grep -v jpeg | xargs -I{} mv {} {}.jpeg


cd ../hindi-train
cat ../ud_hindi_train | xargs -I{} -P 20 wget -q {}
ls | grep -v jpeg | xargs -I{} mv {} {}.jpeg 


cd ../hindi-test
cat ../ud_hindi_test | xargs -I{} -P 20 wget -q {}
ls | grep -v jpeg | xargs -I{} mv {} {}.jpeg


cd ../math-train
cat ../ud_math_train | xargs -I{} -P 20 wget -q {}
ls | grep -v jpeg | xargs -I{} mv {} {}.jpeg


cd ../math-test
cat ../ud_math_test | xargs -I{} -P 20 wget -q {}
ls | grep -v jpeg | xargs -I{} mv {} {}.jpeg


