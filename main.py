import math
import random
import csv
cla_all_num = 0
cla_num = {}
cla_tag_num = {}
landa = 0.6  # 拉普拉斯修正值


def train(taglist, cla):  # 训练，每次插入一条数据
    # 插入分类
    global cla_all_num
    cla_all_num += 1
    if cla in cla_num:  # 是否已存在该分类
        cla_num[cla] += 1
    else:
        cla_num[cla] = 1
    if cla not in cla_tag_num:
        cla_tag_num[cla] = {}  # 创建每个分类的标签字典
    # 插入标签
    tmp_tags = cla_tag_num[cla]  # 浅拷贝，用作别名
    for tag in taglist:
        if tag in tmp_tags:
            tmp_tags[tag] += 1
        else:
            tmp_tags[tag] = 1


def P_C(cla):  # 计算分类 cla 的先验概率
    return cla_num[cla] / cla_all_num


def P_W_C(tag, cla):  # 计算分类 cla 中标签 tag 的后验概率
    tmp_tags = cla_tag_num[cla]  # 浅拷贝，用作别名
    if tag not in cla_tag_num[cla]:
        return landa / (cla_num[cla] + len(tmp_tags) * landa)  # 拉普拉斯修正
    return (tmp_tags[tag] + landa) / (cla_num[cla] + len(tmp_tags) * landa)


def test(test_tags):  # 测试
    res = ''  # 结果
    res_P = None
    for cla in cla_num.keys():
        log_P_W_C = 0
        for tag in test_tags:
            log_P_W_C += math.log(P_W_C(tag, cla), 2)
        tmp_P = log_P_W_C + math.log(P_C(cla), 2)  # P(w|Ci) * P(Ci)
        if res_P is None:
            res = cla
            res_P = tmp_P
        if tmp_P > res_P:
            res = cla
            res_P = tmp_P
    return res, res_P




def beyesi():
    # 训练模型
    i=30000
    with open('adult.csv') as f:
        data = csv.reader(f)
        for x in data:
            train(x[0:14], x[-1])
            i=i-1
            if i <0:
                break;
        f.close()


# 测试模型
# for x in data:
#    print('测试结果：', test(x[0:6]))
if __name__ == '__main__':
    beyesi()  # 创建朴素贝叶斯分类
    # 单例测试模型
    num=0
    right=0
    with open('at.csv') as f:
        data = csv.reader(f)
        for x in data:
            num=num+1
            testcs=x[0:14]
            print(x[-1],test(testcs))

            if(test(testcs)[0] == x[-1]):
                right=right+1
                print(right)
        f.close()
        print(num,right)
        print("正确率为",right/num)
