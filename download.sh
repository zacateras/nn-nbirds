#!/bin/bash

if [ -d "data" ]; then rm -rf "data"; fi
mkdir "data"

wget "https://www.dropbox.com/s/fi2g3zxsn0pdmn1/nbirds.zip" -O $PWD"/data/nbirds.zip"
unzip -q "data/nbirds.zip" -d "data"
rm "data/nbirds.zip"