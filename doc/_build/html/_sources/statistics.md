# Statistics Questions

## Medium Questions

### Question 1: Probability Density Function (Medium)  
The height of adult males in a population follows a normal distribution with a mean of 70 inches and a standard deviation of 3 inches.  
- What is the probability that a randomly selected male is taller than 74 inches?  
- What is the height range that covers the central 95% of the population?  

**Answer:**  

#### Step 1: Probability Calculation  
We need to calculate \(P(X > 74)\).  
- Calculate the Z-score:  
  $
  Z = \frac{X - \mu}{\sigma} = \frac{74 - 70}{3} = \frac{4}{3} \approx 1.33
  $  
- Using the standard normal distribution table,  
  $
  P(Z > 1.33) = 1 - P(Z \leq 1.33) = 1 - 0.9082 = 0.0918
  $  
- Therefore, the probability of being taller than 74 inches is **0.0918** (or 9.18%).  

#### Step 2: Central 95% Range  
The central 95% range corresponds to approximately 1.96 standard deviations from the mean.  
$
\text{Range} = [70 - 1.96 \times 3, 70 + 1.96 \times 3] = [64.12, 75.88]
$  
Hence, the central 95% of male heights lies between **64.12 inches** and **75.88 inches**.  

---

### Question 2: Binomial Distribution (Medium)  
A software company estimates that 80% of its updates are successfully installed on the first attempt. Out of 10 installations, what is the probability that exactly 8 are successful?  

**Answer:**  

#### Step 1: Binomial Probability Formula  
Given:  
- \(n = 10\) (number of trials)  
- \(p = 0.8\) (success probability)  
- \(k = 8\) (number of successes)  

$
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
$  
$$
P(X = 8) = \binom{10}{8} (0.8)^8 (0.2)^2
$  
$$
P(X = 8) = \frac{10 \times 9}{2} \times 0.1678 \times 0.04 = 45 \times 0.0067 = 0.301
$  
Therefore, the probability that exactly 8 installations are successful is **0.301**.  

---

### Question 3: Chi-Square Test for Independence (Medium)  
A survey was conducted to determine the preference for three brands (A, B, C) among two age groups (Youth, Adult). The data is as follows:  

| Age Group | Brand A | Brand B | Brand C |
|----------|--------|--------|--------|
| Youth    |   20   |   30   |   50   |
| Adult    |   25   |   35   |   40   |

Perform a Chi-Square test for independence at a 5% significance level.  

**Answer:**  

#### Step 1: Observed Values (O)  
|         | A  | B  | C  | Total |
|---------|----|----|----|-------|
| Youth   | 20 | 30 | 50 | 100   |
| Adult   | 25 | 35 | 40 | 100   |
| Total   | 45 | 65 | 90 | 200   |

#### Step 2: Expected Values (E)  
$
E = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}}
$  
For Youth-Brand A:  
$
E = \frac{100 \times 45}{200} = 22.5
$  

#### Step 3: Chi-Square Statistic  
$
\chi^2 = \sum \frac{(O - E)^2}{E}
$  
$
\chi^2 = \frac{(20 - 22.5)^2}{22.5} + \frac{(30 - 32.5)^2}{32.5} + \frac{(50 - 45)^2}{45} + \ldots = 2.04
$  

#### Step 4: Decision  
With 2 degrees of freedom, the critical value at 5% significance is 5.99. Since 2.04 < 5.99, we **fail to reject the null hypothesis** (no significant association).  

---

### Question 4: Correlation and Linear Regression (Medium)  
You collected the following paired data points for study hours (X) and test scores (Y):  
\[(2, 50), (3, 60), (4, 70), (5, 80), (6, 90)\]  
Calculate the Pearson correlation coefficient and the linear regression equation.  

**Answer:**  

#### Step 1: Correlation Coefficient (r)  
$
r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}
$  
$
r = 1 \, \text{(perfect positive correlation)}
$  

#### Step 2: Linear Regression Equation  
$
Y = mX + c
$

$
m = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2} = 10
$  
$
c = \bar{Y} - m \bar{X} = 30
$  
Equation:  
$
Y = 10X + 30
$  

---

### Question 5: Poisson Distribution (Medium)  
The number of website visits per minute follows a Poisson distribution with an average rate of 3 visits per minute. What is the probability that exactly 5 visits occur in a given minute?  

**Answer:**  

#### Step 1: Poisson Formula  
$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$  
$
P(X = 5) = \frac{3^5 \times e^{-3}}{5!} = \frac{243 \times 0.0498}{120} = 0.1008
$  
The probability of exactly 5 visits in a minute is **0.1008**.