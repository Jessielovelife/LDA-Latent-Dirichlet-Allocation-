#coding:utf-8
#将大文本文件分割成多个小文本文件
import os

sourceFileName = "baidumerge.txt" #定义要分割的文件
def cutFile():
    print (u"正在读取文件...")
    sourceFileData = open(sourceFileName, 'r', encoding='utf-8')
    ListOfLine = sourceFileData.read().splitlines()#将读取的文件内容按行分割，然后存到一个列表中
    n = len(ListOfLine)
    print (u"文件共有"+str(n)+u"行")
    print (u"请输入需要将文件分割的个数:")
    m = 4
    p = int(n/m + 1)
    print (u"需要将文件分成"+str(m)+u"个子文件")
    print (u"每个文件最多有"+str(p)+u"行")
    print (u"开始进行分割···")
    for i in range(m):
        print (u"正在生成第"+str(i+1)+u"个子文件")
        destFileName = os.path.splitext(sourceFileName)[0]+"_part"+str(i)+".txt" #定义分割后新生成的文件
        destFileData = open(destFileName,"w", encoding='utf-8')
        if(i==m-1):
            for line in ListOfLine[i*p:]:
                destFileData.write(line+'\n')
        else:
            for line in ListOfLine[i*p:(i+1)*p]:
                destFileData.write(line+'\n')
        destFileData.close()
    print (u"分割完成")

cutFile()
