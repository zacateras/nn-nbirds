#!/bin/bash

if [ -d "$DIRECTORY" ]; then
  rmdir "data"
fi

mkdir "data"

wget "https://www.dropbox.com/s/fi2g3zxsn0pdmn1/nbirds.zip" -O $PWD"/data/nbirds.zip"
tar -C "data" -xzf "data/nbirds.zip"
rm "data/nbirds.zip"