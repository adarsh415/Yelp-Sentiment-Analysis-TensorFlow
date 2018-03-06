import data_util
import csv

def batch(batchsize):
    df = data_util.create_dataFrame('yelp.csv')
    text_df = df.as_matrix(columns=['text', 'stars'])
    n_iter=df.shape[0]//batchsize
    start=0
    while n_iter > 0:

        yield text_df[start:start+batchsize,0],text_df[start:start+batchsize,1]
        n_iter -=1
        start +=batchsize



ifile=open('train.csv','r',newline='',encoding="utf-8")
reader=csv.reader(ifile)
ofile=open('train1.csv','w',newline='',encoding="utf-8")
writer=csv.writer(ofile,delimiter='|',quoting=csv.QUOTE_NONE,escapechar='\n')

for row in reader:
    writer.writerow(row)

ifile.close()
ofile.close()

