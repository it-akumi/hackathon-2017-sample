import numpy as np
import argparse
import sys
from scipy.spatial.distance import cosine

def norm(a, b):
    return np.linalg.norm(a - b)

def cos_similar(a, b):
    return cosine(a, b)

def generate_matrix(data, func, output):
    length = len(data)
    mat = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            mat[i][j] = func(data[i], data[j])
    np.savetxt(output, mat, delimiter=',')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="csv data file")
    parser.add_argument("output", help="output matrix data file")
    parser.add_argument("-f", "--func", help="similar function")
    args = parser.parse_args()
    if args.csv:
        data = np.loadtxt(args.csv, delimiter=',')
    else:
        print 'must csv data file'
        exit(-1)

    if not args.csv:
        print 'must csv data file'
        exit(-1)


    func = norm
    if args.func == 'norm':
        func = norm
    else:
        func = cosine

    print 'generate matrix'
    generate_matrix(data, func, args.output)
    print 'done'


if __name__ == "__main__":
   main()