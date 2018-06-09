#!/bin/bash

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
lists_dir="$PWD/data_tools"

for subset in train test validation; do
    copy_from_list "$lists_dir/$subset.txt" \
		   "$dataset_dir" \
		   "${dataset_dir}_$subset"
done
