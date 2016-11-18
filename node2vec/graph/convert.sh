#!/bin/bash

touch $2
sed 's/,/ /g' $1 > $2
