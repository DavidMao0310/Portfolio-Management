# Portfolio-Management
## Modern Portfolio Theory

* Modern portfolio theory (MPT) is a theory on how risk-averse investors can construct portfolios to maximize expected return based on a given level of market risk.

* MPT can also be used to construct a portfolio that minimizes risk for a given level of expected return.

* Modern portfolio theory is very useful for investors trying to construct efficient portfolios using ETFs(***An exchange traded fund (ETF) is a basket of securities that trade on an exchange, just like a stock.***).

* MPT shows that an investor can construct a portfolio of multiple assets that will maximize returns for a given level of risk. Likewise, given a desired level of expected return, an investor can construct a portfolio with the lowest possible risk.

Our first step towards calculating modern portfolio theory portfolios is to get daily and monthly returns. 
Eventually we're going to get the best portfolios of each month based on the **Sharpe ratio**. 

The easiest way to do this is to put all our stock prices into one DataFrame, then to **resample them to the daily and monthly time frames.** We need daily price changes to calculate volatility, which we will use as our measure of risk. In MPT, we quantify risk via volatility.

We'll now generate portfolios to find each month's best one.
`numpy.random.random()` generates random numbers from a uniform distribution, then we **normalize them so they sum to 1** using the `/=` operator. We use these **weights(on each asset)** to calculate returns and volatility. 

Returns are sums of weights times individual returns. Volatility is more complex, and involves the covariances of the different stocks.


### Methodology for MPT
```math
* Price at time $t$: $P_t$
* Daily Return: $r(t)=\frac{P_t-P_{t-1}}{P_{t-1}}$
* Daily Log Return: $log(1+R_t)=logP_t-logP_{t-1}$
* Assest $i$'s weight: $w_i$

We are going to creat many kinds of portfolios, then we loop all of them and find the best one by estimating total return and volatility. 

For $j^{th}$ portfolio, we denote the return on $i^{th}$ asset by $R_{ij}$ and $R_{P_j}$ for portfolio. Suppose $j^{th}$ portfolio consists of $N$ assets, then we have the weight vector and the Return vector$$\begin{bmatrix}w_1\\ w_2\\ \vdots \\ w_N\end{bmatrix}$$
Also, the Return vector is $$\begin{bmatrix}R_{1j}\\ R_{2j}\\ \vdots \\ R_{Nj}\end{bmatrix}$$
Then $$R_{P_j}=\begin{bmatrix}R_{1j}\\ R_{2j}\\ \vdots \\ R_{Nj}\end{bmatrix} \cdot \begin{bmatrix}w_1\\ w_2\\ \vdots \\ w_N\end{bmatrix} \\
=\begin{bmatrix}w_1&w_2&\cdots &w_N\end{bmatrix} \begin{bmatrix}R_{1j}\\ R_{2j}\\ \vdots \\ R_{Nj}\end{bmatrix}\\=\sum_{i=1}^{N}w_iR_{ij}$$
The expected return is also a weighted average of the expected returns on the individual assets, given by $$E(R_{P_j})=E(\sum_{i=1}^{N}w_iR_{ij}) \\ =\sum_{i=1}^{N}w_iE(R_{ij})=\begin{bmatrix}w_1&w_2&\cdots &w_N\end{bmatrix} \begin{bmatrix}E(R_{1j})\\ E(R_{2j})\\ \vdots \\ E(R_{Nj})\end{bmatrix}\\$$
The volatility is computing by standard deviation on return of asset. Hence, the variance on a portfolio is $$\sigma_{P_j}^2=E(R_{P_j}-E({R_{P_j}}))^2\\=\begin{bmatrix}w_1&w_2&\cdots &w_N\end{bmatrix} \begin{bmatrix}\sigma_{1j}^2&\sigma_{1j,2j}&\cdots&\sigma_{1j,Nj}\\ \sigma_{2j,Nj}&\sigma_{2j}^2&\cdots&\sigma_{2j,Nj}\\ \vdots&\vdots&\ddots&\vdots \\ \sigma_{Nj,1j}&\sigma_{Nj,2j}&\cdots&\sigma_{Nj}^2\end{bmatrix}  \begin{bmatrix}w_1\\ w_2\\ \vdots \\ w_N\end{bmatrix}\\==\begin{bmatrix}w_1&w_2&\cdots &w_N\end{bmatrix} \textbf{COV(Pj)}\begin{bmatrix}w_1\\ w_2\\ \vdots \\ w_N\end{bmatrix}\\$$
Where $\textbf{COV(Pj)}$ is the covariance matrix of $j^{th}$ portfolio, then it is easy to get the the volatility of portfolio $\sigma_{P_j}$.

####Sharpr Ratio
Sharpe ratio is used to help investors understand the return of an investment compared to its risk. The ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk. Volatility is a measure of the price fluctuations of an asset or portfolio.
$$Sharpe Ratio = \frac{R_P-r_f}{\sigma_P}$$
Where $R_P$ is the return on portfolio, $r_f$ is the risk-free rate, $\sigma_{P}$ is the volatility of portfolio.
If we get higher sharpe ratio, then better portfolio. 

```

