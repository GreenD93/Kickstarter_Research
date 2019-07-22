import pandas as pd
from scipy import stats

def check_normalD(df,dcolumn,vcolumn):
    # H0 : 정규분포를 따른다.
    # H1 : 정규분포를 따르지 않는다.
    
    group_list = df[dcolumn].unique().tolist()
    group1 = df[df[dcolumn] == group_list[0]][vcolumn].values
    group2 = df[df[dcolumn] == group_list[1]][vcolumn].values
    
    wilcoxon = 0
    if len(group1) < 30 or len(group2) < 30:
        print('---------------------------------------------')
        print('Shapiro 정규성 검정')
        norm_test1 = stats.shapiro(group1)
        norm_test2 = stats.shapiro(group2)
        print("group1 : {0}".format(norm_test1))
        print("group2 : {0}".format(norm_test2))

        
        if norm_test1[1] < 0.05 or norm_test2 < 0.05:
            print('=> H1 : 두 집단 중 정규분포를 따르지 않는 집단이 있다.')
            wilcoxon = 1
        else:
            print('=> H0 :두 집단 모두 정규분포를 따른다.')
        return group1, group2, wilcoxon
    else:
        print('---------------------------------------------')
        print('중심극한정리에 따라서 두 분포 모두 정규분포를 따른다.')
    return group1, group2, wilcoxon

def check_homo(group1,group2):
    # H0 : 두 집단의 분산은 같다.
    # H1 : 두 집단의 분산은 다르다.
    print('---------------------------------------------')
    print('Levene 등분산성 검정')
    homo_test = stats.levene(group1,group2)
    print(homo_test)
    welch = 0
    if homo_test[1] < 0.05:
        print('=> H1 : 두 집단의 분산은 다르다')
        welch = 1
    else:
        print('=> H0 : 두 집단의 분산은 같다')
    return welch

def Ttest(group1,group2,welch):
    # H0 : 두 집단의 평균은 같다.
    # H1 : 두 집단의 평균은 다르다.
    print('---------------------------------------------')
    print('T-test')
    T_test = stats.ttest_ind(group1, group2, equal_var=welch)
    print(T_test)
    if T_test[1] < 0.05:
        print('Welcht-test')
        print('=> H1 : 두 집단의 평균은 다르다')
    else:
        print('독립표본 t-검정')
        print('=> H0 : 두 집단의 평균은 같다')

def Wilcoxon(group1,group2):
    # H0 : 두 집단의 평균은 같다.
    # H1 : 두 집단의 평균은 다르다.
    print('---------------------------------------------')
    print('Wilcoxon rank-sum test')
    Wil_test = stats.ranksums(group1,group2)
    print(Wil_test)
    if Wil_test[1] < 0.05:
        print('=> H1 : 두 집단의 평균은 다르다')
    else:
        print('=> H0 : 두 집단의 평균은 같다')

def T_test(df,dcolumn,vcolumn):
    group1, group2, wilcoxon = check_normalD(df,dcolumn,vcolumn)
    
    if wilcoxon == 1:
        wilcoxon_result =Wilcoxon(group1,group2)
        return wilcoxon_result
    
    welch = check_homo(group1,group2)
    Ttest_result = Ttest(group1,group2,welch)
    return Ttest_result
