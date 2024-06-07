# God View

传统相对论的视角。

约定：
$$
\begin{align}
\vec{\beta}&=\cfrac{\vec{v}}{c}\\
\gamma&=\cfrac{1}{\sqrt{1-\beta^2}}
\end{align}
$$
根据洛伦兹变换，如果$K'$系相对$K$系以速度$\vec{v}$运动，那么两个事件时空间隔的变换为：
$$
\begin{align}
c\Delta t'&=\gamma(c\Delta t-\vec{\beta}\cdot\vec{\Delta x})\\
\vec{\Delta {x'}_{\parallel}}&=\gamma(\vec{\Delta x_{\parallel}}-{\vec\beta}c\Delta t)\\
\vec{\Delta {x'}_{\perp}}&=\vec{\Delta x_{\perp}}\\
\end{align}
$$

### 1. 寻找需要渲染的事件

我们看到的某时刻的图像，就是说明时空坐标上该时刻，某个位置上有东西。如果考虑这个事件和相机的时空事件的差，一定有$\Delta t=0$。（非相对论情形）

那么在相机运动的时候，看到的就是$\Delta t'=0$的事件集合，即要求解方程$c\Delta t=\vec{\beta}\cdot\vec{\Delta x}$，由于$\beta<1$，物体运动速度小于光速时一定只有唯一解（假设延拓到$t\in(-\infty,+\infty)$上），可以通过牛顿迭代法求解这个解的位置。

### 2. 坐标变换

坐标平行和垂直分量可以记作（$\hat{\beta}$表示单位向量）：
$$
\begin{align}
\vec{\Delta x'_{\parallel}}=(\vec{\Delta x'}\cdot\hat{\beta})\hat\beta\\
\vec{\Delta x'_{\perp}}=\vec{\Delta x'}-\vec{\Delta x'_{\parallel}}
\end{align}
$$
代入洛伦兹变换，总体写成矢量形式为：
$$
\begin{align}
\vec{\Delta x'}&=\vec{\Delta x}+(\gamma-1)(\vec{\Delta x'}\cdot\hat{\beta})\hat\beta-\gamma c\Delta t\vec{\beta}\\
&=\vec{\Delta x}+\cfrac{\gamma-1}{\beta^2}(\vec{\Delta x'}\cdot\vec{\beta})\vec\beta-\gamma c\Delta t\vec{\beta}\\
&=\vec{\Delta x}+\cfrac{\gamma^2}{\gamma+1}(\vec{\Delta x'}\cdot\vec{\beta})\vec\beta-\gamma c\Delta t\vec{\beta}\\
\end{align}
$$

### 3. 目前的问题

由于$\Delta t$可能大于零，需要先走一遍模拟把所有时间的write geometry做完，再进行相机运动和观察。

