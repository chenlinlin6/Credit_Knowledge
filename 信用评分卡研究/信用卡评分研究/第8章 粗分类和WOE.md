[TOC]

# 前言

WOE(weight of evidence)称为**权重证据转换**，可以将logistic模型转换为标准评分卡格式。引入WOE转换是为了增加模型的稳定性，减小模型的复杂度。WOE转换又被称为粗分类。

# 1. 基本定义

## 名义变量

对于名义变量来说，需要计算出某个变量各个类别的好坏样本的分布占比，就可以计算出对应类别的woe值。

以下面的**ResStatus**为例子，阐明woe的计算方式。

![image-20200204182642890](https://tva1.sinaimg.cn/large/006tNbRwgy1gblwha5p7sj312c0c040j.jpg)

这里的「正常的分布」是指某个类别下正常的样本数占所有正常样本数的百分比。比如以「屋主」这个类别为例子，正常的个数为365个，总的正常数为803，因此

$$正常的分布=\frac{365}{803}=0.455$$

**WOE**的计算公式为

$$WOE=ln\dfrac{bad\ distribution}{good\ distribution}$$

这里的好坏的分子分母顺序并不是一定是上面写的，这里主要是凸显出该类别下好坏样本差异的情况。当$\dfrac{bad\ distribution}{good\ distribution}>1$的时候为正值，当$\dfrac{bad\ distribution}{good\ distribution}<1$的时候为负值。

如果变量经过woe化以后，该变量对应的模型参数正好是1.

如下图所示，运行下面的代码，对「ResStatus」这个变量进行woe化以后，对应每个类都有相应的woe值。

![image-20200205220425869](https://tva1.sinaimg.cn/large/006tNbRwgy1gblwh90u0zj315w0sojwt.jpg)

对应的模型参数如下：

![image-20200205220613479](https://tva1.sinaimg.cn/large/006tNbRwgy1gblwh86sizj312s0u00yc.jpg)

这里的模型参数为1，代表是概率比的对数，而不是指数的对数。

# 2. 证据权重与标准评分卡

用woe转换，可以将名义变量x经过woe化以后的值表示为：

$$WOE(x)=\delta_1WOE_1+\delta_2WOE_2+\delta_3WOE_3$$

其中$WOE_i$是类别$i$下面的WOE值，$\delta_i$是哑变量，也就是取值为0或者1，当名义变量x为类别i的时候，对应的$\delta_i=1$，其他$\delta$为0。

如果我们将新的转换变量应用到$logistic$模型当中的话，就相当于每个二元变量$\delta_i$都对应着一个唯一的模型参数，而这些参数就代表着对模型「违约/正常」的对数的变化大小。

具体以下面这个例子为例。

假如有三个名义变量$x_1, x_2, x_3$，其「违约/正常」的比率的对数可以表示为：

$$ln(odds)=ln\dfrac{p}{1-p}=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_3$$

假如说各个名义变量的分类如下：

![image-20200205222906758](https://tva1.sinaimg.cn/large/006tNbRwgy1gblwh70vdcj310o0c0756.jpg)

假如说$x_1, x_2, x_3$经过woe转换后得到的变量是$\upsilon_1, \upsilon_2, \upsilon_3$，得到的结果为：

$$ln(odds)=ln\dfrac{p}{1-p}=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_3=\beta_0+\beta_1\upsilon_1+\beta_2\upsilon_2+\beta_3\upsilon_3$$

进一步把$\upsilon_1, \upsilon_2, \upsilon_3$的值代入进去，可以得到

![image-20200205223024826](https://tva1.sinaimg.cn/large/006tNbRwgy1gblwh1ewt3j312s0kg0vs.jpg)

这个形式就是标准评分卡的形式，这里代表：

1. 每个标准评分卡都有个基础得分，在这里就是$\beta_0$。
2. 如果$x_1$取$x_{11}$，那么对应增加的得分就是$\beta_1w_{11}$，其他以此类推。
3. 各个变量取对应类别的值，就增加相应的分数，最终的得分是基础得分加上各个变量的对应激活类别的得分。
4. 最终的得分对应的就是「违约/正常」的比率的对数值。

# 3. 连续变量的WOE值

一般来说，连续型变量与名义变量不同，需要对其进行分段后，计算WOE值，而分段的依据就是分段以后使得该变量对模型的预测性能最强，而其中之一就是应用决策树算法。这种方法叫做最优分段。

如果是追求业务的可解释性，也可以选择简单的等距分箱，比如对年龄这个变量。

分箱以后计算woe值，一般来说，按照变量的分数段排序后，期待对应的woe值是线性或者是单调的。这个也是和评分卡的计算逻辑一致。

![image-20200204190723592](https://tva1.sinaimg.cn/large/006tNbRwgy1gblwh3nstyj30u00ugadb.jpg)

如图呈现出来的是「CustAge」和「TmAtAddress」这两个变量的WOE值和分箱的关系。

可以看得出来「TmAtAddress」并没有严格遵循单调性。

这种情况下可能有两种原因要考虑：

1.一种情况可能是该变量的解释性或者预测能力不能被逻辑回归模型所解释，不适合用逻辑回归模型。这种情况下可以尝试转变分箱的方式，合并等等，如果尝试了多种分箱后仍不能够得到预测能力的单调性，那么至少要排除这个模型。

2.业务理解上解释得通。这种情况下往往是中间的违约率高，极端高或者极端低的违约率低，比如年龄，这种情况下在业务上可以解释为中间的人就业率最高，因此违约率最低。

