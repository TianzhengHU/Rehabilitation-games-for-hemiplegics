import operator
import pprint
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as st
import sklearn.linear_model as lm
import pymysql      
import pandas as pd 


#链接mysql数据库
connect = pymysql.connect(host="localhost", user="root", password="mashtz0303", database="PDSys",charset="utf8mb4")
# connect = pymysql.connect(host="localhost", user="bdi_gj", password="root", database="PDSys",charset="utf8mb4")
cursor=connect.cursor()
#查询数据
#首先要获取数据库中的data值，有多少个pid-- fetchone()方法返回一行数据，fetchall()方法获取所有数据：result=cursor.fetchone()
cursor=connect.cursor(pymysql.cursors.DictCursor)
cursor.execute("SELECT * FROM Patients")
patientslist=cursor.fetchall()#所有patients
pnumber = len(patientslist)#病人的数量

print(patientslist)

#拿到所有病人的pid并且生成一个list
pidlist=[]
for i in range(0,pnumber):
    getpid=operator.itemgetter(i)#拿到所有result的第i个病人
    pid = getpid(patientslist).get("Pid")#拿到第i个病人的pid
    pidlist.append(pid)
print("pidlist")
print(pidlist)


#要拿到每个pid对应的datas
for i in range(0,pnumber):
    xline = []
    H = []
    M = []
    L = []
    FT = []
    N = []
    Acc = []
    pid = pidlist[i]

    #对于每一个pid，拿到所有该pid下的datas
    cursor.execute("SELECT * FROM Datas WHERE Pid='"+ pid + "'")
    dataslist = cursor.fetchall()  # 所有datas
    print("dataslist")
    print(dataslist)
    if len(dataslist)<1:
        continue

    for j in range(0, len(dataslist)):
        xline.append(j+1)
        getdata=operator.itemgetter(j)#获取第j个data数据
        H.append(getdata(dataslist).get("H"))
        M.append(getdata(dataslist).get("M"))
        L.append(getdata(dataslist).get("L"))
        FT.append(getdata(dataslist).get("fastTime"))
        N.append(getdata(dataslist).get("N"))
        Acc.append(getdata(dataslist).get("Acc"))

    print()
    print("start")
    print(j)


    # ----------------------------------开始对H----------------------------------------
    x = np.array(xline)
    y = np.array(H)

    Xt = np.array([np.ones(len(x)), x])
    X = Xt.transpose()  # 转置函数
    Y = y.transpose()
    Z = np.matmul(Xt, X)  # 矩阵乘法，矩阵a乘以矩阵b，生成a * b
    Zinv = np.linalg.inv(Z)  # 矩阵求逆
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)

    # Z3 is the pseudoinverse-伪逆矩阵
    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b
    print("Problem 3a: ""y=", m, "x+", b)

    x2 = x[0:(len(x) - 1)]
    y2 = y[0:(len(x) - 1)]

    Xt = np.array([np.ones(len(x) - 1), x2])
    X = Xt.transpose()
    Y = y2.transpose()
    Z = np.matmul(Xt, X)
    Zinv = np.linalg.inv(Z)
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)
    b = Z3[0]
    m = Z3[1]
    ypred2 = m * x2 + b  # 直接计算矩阵

    print("Problem 3c: ""y=", m, "x+", b)
    y9 = m * x[len(x) - 1] + b
    q = (y9 - y[(len(x) - 1)]) ** 2
    print("predicted y9", y9, "true y9", y[len(x) - 1], "squared test error", q)

    # We create the model.
    lr = lm.LinearRegression()
    # We train the model on our training dataset.
    lr.fit(x[:, np.newaxis], y)
    # Now, we predict points with our trained model.
    y_lr = lr.predict(x[:, np.newaxis])  # y:目标结果 x:因变量

    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b

    print()
    print()
    print()
    print("------------------Linear regression------------------")
    print("Problem 3d: ""y=", lr.coef_, "x+", lr.intercept_)

    getlist = operator.itemgetter(0)
    pid = pidlist[i]
    print(type(pid))
    abbr = "H"
    pprint.pprint("mmmmm")
    m = str(getlist((lr.coef_).tolist()))
    print(type(m))

    pprint.pprint("bbbb")
    print((lr.intercept_))
    print(type(lr.intercept_))
    b = str(np.round((lr.intercept_),5))

    #检查是否已经添加过该病人的该属性数据
    cursor.execute("SELECT * FROM equation WHERE Pid='"+pid+"' AND abbr='"+abbr+"'")
    isExit = len(cursor.fetchall())  # 是否相等
    print("isExit")
    print(isExit)
    sql = "INSERT INTO equation(Pid,Abbr,m,b) VALUES ('" + pid + "','" + abbr + "'," + m + "," + b + ")"

    if isExit>0:#之前已经插入过
        sql="UPDATE equation SET m=" +m+",b=" +b+"  WHERE Pid='" +pid+"' AND abbr='" +abbr+"'"
    # 插入数据
    try:
        # 插入数据cursor.close()
        cursor.execute(sql)
        connect.commit()
        print("插入成功")
        pass
    except Exception as e:
        connect.rollback()
        print("插入失败", e)

# ----------------------------------开始对H----------------------------------------
    # Mileage 横坐标
    # x = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    x = np.array(xline)

    # Tire tread depth 纵坐标
    # y = np.array([394.33, 329.50, 291.00, 255.17, 229.33, 204.83, 179.00, 163.83, 150.33])
    y = np.array(H)

    # Xt = np.array([np.ones(9),x])#np.ones()返回指定形状的新数组
    Xt = np.array([np.ones(len(x)), x])
    X = Xt.transpose()  # 转置函数
    Y = y.transpose()
    Z = np.matmul(Xt, X)  # 矩阵乘法，矩阵a乘以矩阵b，生成a * b
    Zinv = np.linalg.inv(Z)  # 矩阵求逆
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)

    # Z3 is the pseudoinverse-伪逆矩阵
    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b
    print("Problem 3a: ""y=", m, "x+", b)

    # take the first 8 elements
    # x2 = x[0:8]
    # y2 = y[0:8]
    x2 = x[0:(len(x) - 1)]
    y2 = y[0:(len(x) - 1)]

    # Xt = np.array([np.ones(8),x2])
    Xt = np.array([np.ones(len(x) - 1), x2])
    X = Xt.transpose()
    Y = y2.transpose()
    Z = np.matmul(Xt, X)
    Zinv = np.linalg.inv(Z)
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)
    b = Z3[0]
    m = Z3[1]
    ypred2 = m * x2 + b  # 直接计算矩阵

    print("Problem 3c: ""y=", m, "x+", b)
    # y9=m*x[8]+b
    # q=(y9-y[8])**2
    # print("predicted y9",y9,"true y9", y[8], "squared test error",q)
    y9 = m * x[len(x) - 1] + b
    q = (y9 - y[(len(x) - 1)]) ** 2
    print("predicted y9", y9, "true y9", y[len(x) - 1], "squared test error", q)

    # We create the model.
    lr = lm.LinearRegression()
    # We train the model on our training dataset.
    lr.fit(x[:, np.newaxis], y)
    # Now, we predict points with our trained model.
    y_lr = lr.predict(x[:, np.newaxis])  # y:目标结果 x:因变量

    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b

    print()
    print()
    print()
    print("------------------Linear regression------------------")
    print("Problem 3d: ""y=", lr.coef_, "x+", lr.intercept_)

    getlist = operator.itemgetter(0)
    pid = pidlist[i]
    print(type(pid))
    abbr = "H"
    pprint.pprint("mmmmm")
    m = str(getlist((lr.coef_).tolist()))
    print(type(m))

    pprint.pprint("bbbb")
    print((lr.intercept_))
    print(type(lr.intercept_))
    b = str(np.round((lr.intercept_),5))

    #检查是否已经添加过该病人的该属性数据
    cursor.execute("SELECT * FROM equation WHERE Pid='"+pid+"' AND abbr='"+abbr+"'")
    isExit = len(cursor.fetchall())  # 是否相等
    print("isExit")
    print(isExit)
    sql = "INSERT INTO equation(Pid,Abbr,m,b) VALUES ('" + pid + "','" + abbr + "'," + m + "," + b + ")"

    if isExit>0:#之前已经插入过
        sql="UPDATE equation SET m=" +m+",b=" +b+"  WHERE Pid='" +pid+"' AND abbr='" +abbr+"'"
    # 插入数据
    try:
        # 插入数据cursor.close()
        cursor.execute(sql)
        connect.commit()
        print("插入成功")
        pass
    except Exception as e:
        connect.rollback()
        print("插入失败", e)



# ----------------------------------开始对MMMMMMM----------------------------------------
    # Mileage 横坐标
    # x = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    x = np.array(xline)

    # Tire tread depth 纵坐标
    # y = np.array([394.33, 329.50, 291.00, 255.17, 229.33, 204.83, 179.00, 163.83, 150.33])
    y = np.array(M)

    # Xt = np.array([np.ones(9),x])#np.ones()返回指定形状的新数组
    Xt = np.array([np.ones(len(x)), x])
    X = Xt.transpose()  # 转置函数
    Y = y.transpose()
    Z = np.matmul(Xt, X)  # 矩阵乘法，矩阵a乘以矩阵b，生成a * b
    Zinv = np.linalg.inv(Z)  # 矩阵求逆
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)

    # Z3 is the pseudoinverse-伪逆矩阵
    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b
    print("Problem 3a: ""y=", m, "x+", b)

    # take the first 8 elements
    # x2 = x[0:8]
    # y2 = y[0:8]
    x2 = x[0:(len(x) - 1)]
    y2 = y[0:(len(x) - 1)]

    # Xt = np.array([np.ones(8),x2])
    Xt = np.array([np.ones(len(x) - 1), x2])
    X = Xt.transpose()
    Y = y2.transpose()
    Z = np.matmul(Xt, X)
    Zinv = np.linalg.inv(Z)
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)
    b = Z3[0]
    m = Z3[1]
    ypred2 = m * x2 + b  # 直接计算矩阵

    print("Problem 3c: ""y=", m, "x+", b)
    # y9=m*x[8]+b
    # q=(y9-y[8])**2
    # print("predicted y9",y9,"true y9", y[8], "squared test error",q)
    y9 = m * x[len(x) - 1] + b
    q = (y9 - y[(len(x) - 1)]) ** 2
    print("predicted y9", y9, "true y9", y[len(x) - 1], "squared test error", q)

    # We create the model.
    lr = lm.LinearRegression()
    # We train the model on our training dataset.
    lr.fit(x[:, np.newaxis], y)
    # Now, we predict points with our trained model.
    y_lr = lr.predict(x[:, np.newaxis])  # y:目标结果 x:因变量

    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b

    print()
    print()
    print()
    print("------------------Linear regression------------------")
    print("Problem 3d: ""y=", lr.coef_, "x+", lr.intercept_)

    getlist = operator.itemgetter(0)
    pid = pidlist[i]
    print(type(pid))
    abbr = "M"
    pprint.pprint("mmmmm")
    m = str(getlist((lr.coef_).tolist()))
    print(type(m))

    pprint.pprint("bbbb")
    print((lr.intercept_))
    print(type(lr.intercept_))
    b = str(np.round((lr.intercept_),5))

    #检查是否已经添加过该病人的该属性数据
    cursor.execute("SELECT * FROM equation WHERE Pid='"+pid+"' AND abbr='"+abbr+"'")
    isExit = len(cursor.fetchall())  # 是否相等
    print("isExit")
    print(isExit)
    sql = "INSERT INTO equation(Pid,Abbr,m,b) VALUES ('" + pid + "','" + abbr + "'," + m + "," + b + ")"

    if isExit>0:#之前已经插入过
        sql="UPDATE equation SET m=" +m+",b=" +b+"  WHERE Pid='" +pid+"' AND abbr='" +abbr+"'"
    # 插入数据
    try:
        # 插入数据cursor.close()
        cursor.execute(sql)
        connect.commit()
        print("插入成功")
        pass
    except Exception as e:
        connect.rollback()
        print("插入失败", e)


# ----------------------------------开始对H----------------------------------------
    # Mileage 横坐标
    # x = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    x = np.array(xline)

    # Tire tread depth 纵坐标
    # y = np.array([394.33, 329.50, 291.00, 255.17, 229.33, 204.83, 179.00, 163.83, 150.33])
    y = np.array(H)

    # Xt = np.array([np.ones(9),x])#np.ones()返回指定形状的新数组
    Xt = np.array([np.ones(len(x)), x])
    X = Xt.transpose()  # 转置函数
    Y = y.transpose()
    Z = np.matmul(Xt, X)  # 矩阵乘法，矩阵a乘以矩阵b，生成a * b
    Zinv = np.linalg.inv(Z)  # 矩阵求逆
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)

    # Z3 is the pseudoinverse-伪逆矩阵
    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b
    print("Problem 3a: ""y=", m, "x+", b)

    # take the first 8 elements
    # x2 = x[0:8]
    # y2 = y[0:8]
    x2 = x[0:(len(x) - 1)]
    y2 = y[0:(len(x) - 1)]

    # Xt = np.array([np.ones(8),x2])
    Xt = np.array([np.ones(len(x) - 1), x2])
    X = Xt.transpose()
    Y = y2.transpose()
    Z = np.matmul(Xt, X)
    Zinv = np.linalg.inv(Z)
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)
    b = Z3[0]
    m = Z3[1]
    ypred2 = m * x2 + b  # 直接计算矩阵

    print("Problem 3c: ""y=", m, "x+", b)
    # y9=m*x[8]+b
    # q=(y9-y[8])**2
    # print("predicted y9",y9,"true y9", y[8], "squared test error",q)
    y9 = m * x[len(x) - 1] + b
    q = (y9 - y[(len(x) - 1)]) ** 2
    print("predicted y9", y9, "true y9", y[len(x) - 1], "squared test error", q)

    # We create the model.
    lr = lm.LinearRegression()
    # We train the model on our training dataset.
    lr.fit(x[:, np.newaxis], y)
    # Now, we predict points with our trained model.
    y_lr = lr.predict(x[:, np.newaxis])  # y:目标结果 x:因变量

    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b

    print()
    print()
    print()
    print("------------------Linear regression------------------")
    print("Problem 3d: ""y=", lr.coef_, "x+", lr.intercept_)

    getlist = operator.itemgetter(0)
    pid = pidlist[i]
    print(type(pid))
    abbr = "H"
    pprint.pprint("mmmmm")
    m = str(getlist((lr.coef_).tolist()))
    print(type(m))

    pprint.pprint("bbbb")
    print((lr.intercept_))
    print(type(lr.intercept_))
    b = str(np.round((lr.intercept_),5))

    #检查是否已经添加过该病人的该属性数据
    cursor.execute("SELECT * FROM equation WHERE Pid='"+pid+"' AND abbr='"+abbr+"'")
    isExit = len(cursor.fetchall())  # 是否相等
    print("isExit")
    print(isExit)
    sql = "INSERT INTO equation(Pid,Abbr,m,b) VALUES ('" + pid + "','" + abbr + "'," + m + "," + b + ")"

    if isExit>0:#之前已经插入过
        sql="UPDATE equation SET m=" +m+",b=" +b+"  WHERE Pid='" +pid+"' AND abbr='" +abbr+"'"
    # 插入数据
    try:
        # 插入数据cursor.close()
        cursor.execute(sql)
        connect.commit()
        print("插入成功")
        pass
    except Exception as e:
        connect.rollback()
        print("插入失败", e)

# ----------------------------------开始对LLLLLLLL----------------------------------
    # Mileage 横坐标
    # x = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    x = np.array(xline)

    # Tire tread depth 纵坐标
    # y = np.array([394.33, 329.50, 291.00, 255.17, 229.33, 204.83, 179.00, 163.83, 150.33])
    y = np.array(L)

    # Xt = np.array([np.ones(9),x])#np.ones()返回指定形状的新数组
    Xt = np.array([np.ones(len(x)), x])
    X = Xt.transpose()  # 转置函数
    Y = y.transpose()
    Z = np.matmul(Xt, X)  # 矩阵乘法，矩阵a乘以矩阵b，生成a * b
    Zinv = np.linalg.inv(Z)  # 矩阵求逆
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)

    # Z3 is the pseudoinverse-伪逆矩阵
    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b
    print("Problem 3a: ""y=", m, "x+", b)

    # take the first 8 elements
    # x2 = x[0:8]
    # y2 = y[0:8]
    x2 = x[0:(len(x) - 1)]
    y2 = y[0:(len(x) - 1)]

    # Xt = np.array([np.ones(8),x2])
    Xt = np.array([np.ones(len(x) - 1), x2])
    X = Xt.transpose()
    Y = y2.transpose()
    Z = np.matmul(Xt, X)
    Zinv = np.linalg.inv(Z)
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)
    b = Z3[0]
    m = Z3[1]
    ypred2 = m * x2 + b  # 直接计算矩阵

    print("Problem 3c: ""y=", m, "x+", b)
    # y9=m*x[8]+b
    # q=(y9-y[8])**2
    # print("predicted y9",y9,"true y9", y[8], "squared test error",q)
    y9 = m * x[len(x) - 1] + b
    q = (y9 - y[(len(x) - 1)]) ** 2
    print("predicted y9", y9, "true y9", y[len(x) - 1], "squared test error", q)

    # We create the model.
    lr = lm.LinearRegression()
    # We train the model on our training dataset.
    lr.fit(x[:, np.newaxis], y)
    # Now, we predict points with our trained model.
    y_lr = lr.predict(x[:, np.newaxis])  # y:目标结果 x:因变量

    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b

    print()
    print()
    print()
    print("------------------Linear regression------------------")
    print("Problem 3d: ""y=", lr.coef_, "x+", lr.intercept_)

    getlist = operator.itemgetter(0)
    pid = pidlist[i]
    print(type(pid))
    abbr = "L"
    pprint.pprint("mmmmm")
    m = str(getlist((lr.coef_).tolist()))
    print(type(m))

    pprint.pprint("bbbb")
    print((lr.intercept_))
    print(type(lr.intercept_))
    b = str(np.round((lr.intercept_),5))

    #检查是否已经添加过该病人的该属性数据
    cursor.execute("SELECT * FROM equation WHERE Pid='"+pid+"' AND abbr='"+abbr+"'")
    isExit = len(cursor.fetchall())  # 是否相等
    print("isExit")
    print(isExit)
    sql = "INSERT INTO equation(Pid,Abbr,m,b) VALUES ('" + pid + "','" + abbr + "'," + m + "," + b + ")"

    if isExit>0:#之前已经插入过
        sql="UPDATE equation SET m=" +m+",b=" +b+"  WHERE Pid='" +pid+"' AND abbr='" +abbr+"'"
    # 插入数据
    try:
        # 插入数据cursor.close()
        cursor.execute(sql)
        connect.commit()
        print("插入成功")
        pass
    except Exception as e:
        connect.rollback()
        print("插入失败", e)

# ----------------------------------开始对NNNNNNNNN---------------------------------------
    # Mileage 横坐标
    # x = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    x = np.array(xline)

    # Tire tread depth 纵坐标
    # y = np.array([394.33, 329.50, 291.00, 255.17, 229.33, 204.83, 179.00, 163.83, 150.33])
    y = np.array(N)

    # Xt = np.array([np.ones(9),x])#np.ones()返回指定形状的新数组
    Xt = np.array([np.ones(len(x)), x])
    X = Xt.transpose()  # 转置函数
    Y = y.transpose()
    Z = np.matmul(Xt, X)  # 矩阵乘法，矩阵a乘以矩阵b，生成a * b
    Zinv = np.linalg.inv(Z)  # 矩阵求逆
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)

    # Z3 is the pseudoinverse-伪逆矩阵
    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b
    print("Problem 3a: ""y=", m, "x+", b)

    # take the first 8 elements
    # x2 = x[0:8]
    # y2 = y[0:8]
    x2 = x[0:(len(x) - 1)]
    y2 = y[0:(len(x) - 1)]

    # Xt = np.array([np.ones(8),x2])
    Xt = np.array([np.ones(len(x) - 1), x2])
    X = Xt.transpose()
    Y = y2.transpose()
    Z = np.matmul(Xt, X)
    Zinv = np.linalg.inv(Z)
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)
    b = Z3[0]
    m = Z3[1]
    ypred2 = m * x2 + b  # 直接计算矩阵

    print("Problem 3c: ""y=", m, "x+", b)
    # y9=m*x[8]+b
    # q=(y9-y[8])**2
    # print("predicted y9",y9,"true y9", y[8], "squared test error",q)
    y9 = m * x[len(x) - 1] + b
    q = (y9 - y[(len(x) - 1)]) ** 2
    print("predicted y9", y9, "true y9", y[len(x) - 1], "squared test error", q)

    # We create the model.
    lr = lm.LinearRegression()
    # We train the model on our training dataset.
    lr.fit(x[:, np.newaxis], y)
    # Now, we predict points with our trained model.
    y_lr = lr.predict(x[:, np.newaxis])  # y:目标结果 x:因变量

    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b

    print()
    print()
    print()
    print("------------------Linear regression------------------")
    print("Problem 3d: ""y=", lr.coef_, "x+", lr.intercept_)

    getlist = operator.itemgetter(0)
    pid = pidlist[i]
    print(type(pid))
    abbr = "N"
    pprint.pprint("mmmmm")
    m = str(getlist((lr.coef_).tolist()))
    print(type(m))

    pprint.pprint("bbbb")
    print((lr.intercept_))
    print(type(lr.intercept_))
    b = str(np.round((lr.intercept_),5))

    #检查是否已经添加过该病人的该属性数据
    cursor.execute("SELECT * FROM equation WHERE Pid='"+pid+"' AND abbr='"+abbr+"'")
    isExit = len(cursor.fetchall())  # 是否相等
    print("isExit")
    print(isExit)
    sql = "INSERT INTO equation(Pid,Abbr,m,b) VALUES ('" + pid + "','" + abbr + "'," + m + "," + b + ")"

    if isExit>0:#之前已经插入过
        sql="UPDATE equation SET m=" +m+",b=" +b+"  WHERE Pid='" +pid+"' AND abbr='" +abbr+"'"
    # 插入数据
    try:
        # 插入数据cursor.close()
        cursor.execute(sql)
        connect.commit()
        print("插入成功")
        pass
    except Exception as e:
        connect.rollback()
        print("插入失败", e)


# ----------------------------------开始对Accccccc-----------------------------------
    # Mileage 横坐标
    # x = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    x = np.array(xline)

    # Tire tread depth 纵坐标
    # y = np.array([394.33, 329.50, 291.00, 255.17, 229.33, 204.83, 179.00, 163.83, 150.33])
    y = np.array(Acc)

    # Xt = np.array([np.ones(9),x])#np.ones()返回指定形状的新数组
    Xt = np.array([np.ones(len(x)), x])
    X = Xt.transpose()  # 转置函数
    Y = y.transpose()
    Z = np.matmul(Xt, X)  # 矩阵乘法，矩阵a乘以矩阵b，生成a * b
    Zinv = np.linalg.inv(Z)  # 矩阵求逆
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)

    # Z3 is the pseudoinverse-伪逆矩阵
    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b
    print("Problem 3a: ""y=", m, "x+", b)

    # take the first 8 elements
    # x2 = x[0:8]
    # y2 = y[0:8]
    x2 = x[0:(len(x) - 1)]
    y2 = y[0:(len(x) - 1)]

    # Xt = np.array([np.ones(8),x2])
    Xt = np.array([np.ones(len(x) - 1), x2])
    X = Xt.transpose()
    Y = y2.transpose()
    Z = np.matmul(Xt, X)
    Zinv = np.linalg.inv(Z)
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)
    b = Z3[0]
    m = Z3[1]
    ypred2 = m * x2 + b  # 直接计算矩阵

    print("Problem 3c: ""y=", m, "x+", b)
    # y9=m*x[8]+b
    # q=(y9-y[8])**2
    # print("predicted y9",y9,"true y9", y[8], "squared test error",q)
    y9 = m * x[len(x) - 1] + b
    q = (y9 - y[(len(x) - 1)]) ** 2
    print("predicted y9", y9, "true y9", y[len(x) - 1], "squared test error", q)

    # We create the model.
    lr = lm.LinearRegression()
    # We train the model on our training dataset.
    lr.fit(x[:, np.newaxis], y)
    # Now, we predict points with our trained model.
    y_lr = lr.predict(x[:, np.newaxis])  # y:目标结果 x:因变量

    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b

    print()
    print()
    print()
    print("------------------Linear regression------------------")
    print("Problem 3d: ""y=", lr.coef_, "x+", lr.intercept_)

    getlist = operator.itemgetter(0)
    pid = pidlist[i]
    print(type(pid))
    abbr = "Acc"
    pprint.pprint("mmmmm")
    m = str(getlist((lr.coef_).tolist()))
    print(type(m))

    pprint.pprint("bbbb")
    print((lr.intercept_))
    print(type(lr.intercept_))
    b = str(np.round((lr.intercept_),5))

    #检查是否已经添加过该病人的该属性数据
    cursor.execute("SELECT * FROM equation WHERE Pid='"+pid+"' AND abbr='"+abbr+"'")
    isExit = len(cursor.fetchall())  # 是否相等
    print("isExit")
    print(isExit)
    sql = "INSERT INTO equation(Pid,Abbr,m,b) VALUES ('" + pid + "','" + abbr + "'," + m + "," + b + ")"

    if isExit>0:#之前已经插入过
        sql="UPDATE equation SET m=" +m+",b=" +b+"  WHERE Pid='" +pid+"' AND abbr='" +abbr+"'"
    # 插入数据
    try:
        # 插入数据cursor.close()
        cursor.execute(sql)
        connect.commit()
        print("插入成功")
        pass
    except Exception as e:
        connect.rollback()
        print("插入失败", e)



# ----------------------------------开始对FTTTTTTTT----------------------------------------
    # Mileage 横坐标
    # x = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32])
    x = np.array(xline)

    # Tire tread depth 纵坐标
    # y = np.array([394.33, 329.50, 291.00, 255.17, 229.33, 204.83, 179.00, 163.83, 150.33])
    y = np.array(FT)

    # Xt = np.array([np.ones(9),x])#np.ones()返回指定形状的新数组
    Xt = np.array([np.ones(len(x)), x])
    X = Xt.transpose()  # 转置函数
    Y = y.transpose()
    Z = np.matmul(Xt, X)  # 矩阵乘法，矩阵a乘以矩阵b，生成a * b
    Zinv = np.linalg.inv(Z)  # 矩阵求逆
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)

    # Z3 is the pseudoinverse-伪逆矩阵
    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b
    print("Problem 3a: ""y=", m, "x+", b)

    # take the first 8 elements
    # x2 = x[0:8]
    # y2 = y[0:8]
    x2 = x[0:(len(x) - 1)]
    y2 = y[0:(len(x) - 1)]

    # Xt = np.array([np.ones(8),x2])
    Xt = np.array([np.ones(len(x) - 1), x2])
    X = Xt.transpose()
    Y = y2.transpose()
    Z = np.matmul(Xt, X)
    Zinv = np.linalg.inv(Z)
    Z2 = np.matmul(Zinv, Xt)
    Z3 = np.matmul(Z2, Y)
    b = Z3[0]
    m = Z3[1]
    ypred2 = m * x2 + b  # 直接计算矩阵

    print("Problem 3c: ""y=", m, "x+", b)
    # y9=m*x[8]+b
    # q=(y9-y[8])**2
    # print("predicted y9",y9,"true y9", y[8], "squared test error",q)
    y9 = m * x[len(x) - 1] + b
    q = (y9 - y[(len(x) - 1)]) ** 2
    print("predicted y9", y9, "true y9", y[len(x) - 1], "squared test error", q)

    # We create the model.
    lr = lm.LinearRegression()
    # We train the model on our training dataset.
    lr.fit(x[:, np.newaxis], y)
    # Now, we predict points with our trained model.
    y_lr = lr.predict(x[:, np.newaxis])  # y:目标结果 x:因变量

    b = Z3[0]
    m = Z3[1]
    ypred = m * x + b

    print()
    print()
    print()
    print("------------------Linear regression------------------")
    print("Problem 3d: ""y=", lr.coef_, "x+", lr.intercept_)

    getlist = operator.itemgetter(0)
    pid = pidlist[i]
    print(type(pid))
    abbr = "fastTime"
    pprint.pprint("mmmmm")
    m = str(getlist((lr.coef_).tolist()))
    print(type(m))

    pprint.pprint("bbbb")
    print((lr.intercept_))
    print(type(lr.intercept_))
    b = str(np.round((lr.intercept_),5))

    #检查是否已经添加过该病人的该属性数据
    cursor.execute("SELECT * FROM equation WHERE Pid='"+pid+"' AND abbr='"+abbr+"'")
    isExit = len(cursor.fetchall())  # 是否相等
    print("isExit")
    print(isExit)
    sql = "INSERT INTO equation(Pid,Abbr,m,b) VALUES ('" + pid + "','" + abbr + "'," + m + "," + b + ")"

    if isExit>0:#之前已经插入过
        sql="UPDATE equation SET m=" +m+",b=" +b+"  WHERE Pid='" +pid+"' AND abbr='" +abbr+"'"
    # 插入数据
    try:
        # 插入数据cursor.close()
        cursor.execute(sql)
        connect.commit()
        print("插入成功")
        pass
    except Exception as e:
        connect.rollback()
        print("插入失败", e)
