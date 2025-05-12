# Statistics Questions

## Conceptual Questions

### Question 1: Mean vs. Median (Easy)  
What is the difference between the mean and the median, and when would you prefer to use the median over the mean?  

**Answer:**  
- **Mean:** The average of all values. Calculated as the sum of values divided by the number of values.  
- **Median:** The middle value when the data is sorted.  
- **When to Use Median:**  
  - The median is preferred when the data is skewed or contains outliers because it is less affected by extreme values.  
  - For example, in income data where a few individuals have significantly higher earnings, the median gives a more representative value than the mean.  

---

### Question 2: Variance and Standard Deviation (Conceptual)  
What do variance and standard deviation measure in a dataset?  

**Answer:**  
- **Variance:** Measures how far data points are spread out from the mean. Calculated as the average of the squared differences from the mean.  
- **Standard Deviation:** The square root of variance, giving a measure of spread in the same units as the data.  
- **Usefulness:**  
  - Both metrics quantify variability, but standard deviation is more interpretable because it has the same unit as the data.  
  - Low variance means data points are close to the mean, while high variance indicates they are spread out.  

---

### Question 3: P-Value (Conceptual)  
What is a p-value, and what does it signify in hypothesis testing?  

**Answer:**  
- The p-value is the probability of obtaining test results at least as extreme as the observed results, assuming that the null hypothesis is true.  
- **Interpretation:**  
  - A low p-value (< 0.05) indicates strong evidence against the null hypothesis, leading to its rejection.  
  - A high p-value (> 0.05) suggests insufficient evidence to reject the null hypothesis.  
- **Important Note:**  
  - A low p-value does not prove that the alternative hypothesis is true; it only suggests that the observed data is unlikely under the null hypothesis.  

---

### Question 4: Confidence Interval (Conceptual)  
What is a confidence interval, and how do you interpret it?  

**Answer:**  
- A confidence interval (CI) is a range of values used to estimate a population parameter with a specified level of confidence (e.g., 95%).  
- **Interpretation:**  
  - A 95% CI means that if the same sampling method is repeated many times, approximately 95% of the calculated intervals will contain the true population parameter.  
- **Example:**  
  - If the CI for the mean height of adults is (170 cm, 180 cm), it suggests that the true mean height is likely within this range.  

---

### Question 5: Type I and Type II Errors (Conceptual)  
What are Type I and Type II errors in hypothesis testing?  

**Answer:**  
- **Type I Error (False Positive):** Rejecting the null hypothesis when it is true.  
  - Example: Concluding that a drug is effective when it actually isn't.  
- **Type II Error (False Negative):** Failing to reject the null hypothesis when it is false.  
  - Example: Concluding that a drug is not effective when it actually is.  
- **Trade-off:**  
  - Reducing the probability of one type of error usually increases the probability of the other.  
  - Significance level (alpha) controls the rate of Type I errors, while power affects Type II errors.  

---

### Question 6: Central Limit Theorem (Conceptual)  
Why is the Central Limit Theorem (CLT) important in statistics?  

**Answer:**  
- The CLT states that the distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the population's original distribution.  
- **Why Important:**  
  - Allows us to make inferences about the population mean using sample data.  
  - Justifies using the normal distribution for hypothesis testing and constructing confidence intervals, especially for large samples.  

---

### Question 7: Correlation vs. Causation (Conceptual)  
What is the difference between correlation and causation?  

**Answer:**  
- **Correlation:** A statistical relationship between two variables, where changes in one variable are associated with changes in another.  
- **Causation:** A relationship where one variable directly affects another.  
- **Key Difference:**  
  - Correlation does not imply causation.  
  - Just because two variables are correlated does not mean one causes the other.  
- **Example:**  
  - Ice cream sales and drowning incidents may be correlated, but the cause is hot weather, not ice cream consumption.  

---

### Question 8: Homoscedasticity (Conceptual)  
What is homoscedasticity, and why is it important in regression analysis?  

**Answer:**  
- **Homoscedasticity:** The variance of errors is constant across all levels of the independent variable.  
- **Importance:**  
  - A key assumption in linear regression.  
  - Violations (heteroscedasticity) can lead to inefficient estimates and affect hypothesis tests.  
- **Detection:**  
  - Residual plots: Plot residuals against predicted values. If the spread of residuals is roughly constant, homoscedasticity is present.  

---

### Question 9: Overfitting vs. Underfitting (Conceptual)  
What is the difference between overfitting and underfitting in machine learning?  

**Answer:**  
- **Overfitting:** The model learns the noise and patterns of the training data too well, performing poorly on new data.  
- **Underfitting:** The model is too simple to capture the underlying pattern in the data.  
- **Balance:**  
  - Overfitting occurs when the model is too complex (e.g., high degree polynomial).  
  - Underfitting occurs when the model is too simple (e.g., linear model for non-linear data).  
- **Solution:**  
  - Use cross-validation, regularization, and simpler models to combat overfitting.  
  - Increase model complexity for underfitting.  
---

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
$
P(X = 8) = \binom{10}{8} (0.8)^8 (0.2)^2
$  
$
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