#!/bin/bash

if [ -d "data" ]; then rm -rf "data"; fi
mkdir "data"

wget "https://www.dropbox.com/s/fi2g3zxsn0pdmn1/nbirds.zip" -O $PWD"/data/nbirds.zip"
unzip -q "data/nbirds.zip" -d "data"
rm "data/nbirds.zip"

## Fixed tvt split

copy_from_list() {
    list="$1"
    src="$2"
    dst="$3"

    while read -r file; do
	mkdir -p "$dst/$(dirname "$file")"
	cp "$src/$file" "$dst/$file"
    done < "$list"
}

dataset_dir="$PWD/data/SET_A"
lists_dir="$PWD/dataset_split"

for subset in train test validation; do
    copy_from_list "$lists_dir/$subset.txt" \
		   "$dataset_dir" \
		   "${dataset_dir}_$subset"
done
