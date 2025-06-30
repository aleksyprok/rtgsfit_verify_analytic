rtgsfit_code_path="/home/alex.prokopyszyn/GitHub/rtgsfit"
this_file_path="$(dirname "$(realpath "$0")")"
constants_c_path="$this_file_path/../constants.c"

# Clean src directory
cd $rtgsfit_code_path/src
make clean

# Copy constants.c to src directory
cp $constants_c_path .

make -f Makefile_no_MDS
