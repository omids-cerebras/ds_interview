# Probability Questions

## Conceptual Questions

## Question 1: Law of Large Numbers (Easy)  
What does the Law of Large Numbers state, and how is it useful in probability?  

**Answer:**  
The Law of Large Numbers states that as the number of trials of a random experiment increases, the average of the observed outcomes approaches the expected value.  
- In practical terms, it means that if you flip a fair coin many times, the proportion of heads will get closer to 0.5 as the number of flips increases.  
- It is useful because it justifies using sample averages to estimate population parameters, especially in statistics and data analysis.  

---

## Question 2: Conditional Probability (Easy)  
Explain conditional probability and provide a real-world example.  

**Answer:**  
Conditional probability is the probability of an event occurring given that another event has already occurred. It is denoted as $P(A \mid B)$.  
- **Formula:**  
  $ 
  P(A \mid B) = \frac{P(A \cap B)}{P(B)}
  $
- **Example:**  
  Suppose 30% of employees in a company are engineers, and 20% are engineers who know Python. The probability that a randomly chosen engineer knows Python is:  
  $
  P(\text{Python} \mid \text{Engineer}) = \frac{0.20}{0.30} = 0.67
  $
  Hence, given that an employee is an engineer, there is a 67% chance they know Python.  

---

## Question 3: Bayes' Theorem (Easy)  
What is Bayes' Theorem, and why is it important?  

**Answer:**  
Bayes' Theorem provides a way to update the probability of a hypothesis based on new evidence.  
- **Formula:**  
  $
  P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}
  $ 
- **Importance:**  
  - Allows us to incorporate new data into our existing beliefs.  
  - Widely used in applications such as spam filtering, medical diagnostics, and machine learning.  

---

## Question 6: Independent Events (Easy)  
What does it mean for two events to be independent?  

**Answer:**  
Two events are independent if the occurrence of one event does not affect the probability of the other event occurring.  
- **Mathematically:**  
   $
  P(A \cap B) = P(A) \times P(B)
   $ 
- **Example:**  
  Rolling a die and flipping a coin are independent events because the result of one does not influence the other.  

---

## Question 7: Random Variable (Easy)  
What is a random variable, and what are its types?  

**Answer:**  
A random variable is a variable that takes numerical values determined by the outcome of a random experiment.  
- **Types:**  
  1. **Discrete Random Variable:** Takes a finite or countable number of values (e.g., number of heads in 10 coin flips).  
  2. **Continuous Random Variable:** Takes an infinite number of possible values within a range (e.g., height of individuals).  

---

## Question 8: Central Limit Theorem (Easy)  
What does the Central Limit Theorem (CLT) state?  

**Answer:**  
The CLT states that the distribution of the sample mean of a large number of independent, identically distributed (i.i.d.) random variables approaches a normal distribution, regardless of the original distribution's shape.  
- **Why it matters:**  
  - Enables statistical inference because we can assume normality when sample sizes are large.  
  - Important for hypothesis testing and constructing confidence intervals.  

---

## Question 9: Simpson's Paradox (Medium)  
What is Simpson's Paradox, and why is it counterintuitive?  

**Answer:**  
Simpson's Paradox occurs when a trend appears in several groups of data but disappears or reverses when the data is combined.  
- **Example:**  
  A medication might show improvement in two separate age groups, but when the data is combined, it appears less effective.  
- **Reason:**  
  The paradox arises due to lurking variables or confounding factors that distort the apparent relationship.  
- **Key Lesson:**  
  Always analyze data carefully before combining groups.  




## Easy Questions

### Question 1: Coin Toss (Easy)  
You flip a fair coin three times. What is the probability of getting exactly two heads?  

**Answer:**  
1. The total number of outcomes when flipping three coins is $2^3 = 8$.  
2. The number of favorable outcomes (two heads) can be calculated using combinations:  
   $
   \text{Number of ways} = \binom{3}{2} = \frac{3!}{2! \times 1!} = 3
   $  
3. Probability:  
   $
   P = \frac{\text{Number of favorable outcomes}}{\text{Total outcomes}} = \frac{3}{8}
   $  

---

### Question 2: Drawing Cards (Easy)  
A standard deck of 52 cards has 4 aces. What is the probability of drawing an ace on the first draw?  

**Answer:**  
1. There are 4 aces in a deck of 52 cards.  
2. The probability of drawing an ace:  
   $
   P = \frac{4}{52} = \frac{1}{13} \approx 0.0769
   $  

---

## Medium Questions

### Question 3: Dice Sum (Medium)  
Two fair six-sided dice are rolled. What is the probability that the sum of the two dice equals 7?  

**Answer:**  
1. Possible pairs that sum to 7:  
   - (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)  
   - There are 6 favorable outcomes.  
2. Total number of outcomes when rolling two dice: $6 \times 6 = 36$  
3. Probability:  
   $
   P = \frac{6}{36} = \frac{1}{6} \approx 0.1667
   $  

---

### Question 4: Defective Items (Medium)  
A factory produces 5% defective items. If you randomly pick 3 items, what is the probability that at least one is defective?  

**Answer:**  
1. Let $A$ be the event that at least one is defective.  
   - The complement of $A$ (no defective items) can be calculated as follows:  
   $
   P(\text{no defective}) = (1 - 0.05)^3 = 0.95^3 \approx 0.857
   $  
2. Using the complement rule:  
   $
   P(\text{at least one defective}) = 1 - P(\text{no defective}) = 1 - 0.857 = 0.143
   $  

---

### Question 5: Birthday Paradox (Medium)  
What is the minimum number of people in a room such that the probability that at least two of them share the same birthday is greater than 50%?  

**Answer:**  
1. Let $n$ be the number of people.  
2. The probability that no one shares the same birthday:  
   $
   P(\text{no shared birthday}) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \times \ldots \times \frac{365 - n + 1}{365}
   $  
3. For $P(\text{at least one shared birthday}) > 0.5$:  
   - Numerical approximation shows that with $n = 23$, the probability exceeds 50%.  
   $
   P(\text{at least one shared birthday}) \approx 1 - 0.4927 = 0.507
   $  
4. **Answer:** At least 23 people are needed.  

---

## Bayes' Rule Problem (Medium)

### Problem Statement:  
A rare disease affects 1 in 1,000 people (prevalence = 0.1%). A medical test designed to detect the disease has the following characteristics:  
- **Sensitivity (True Positive Rate):** 99%  
- **Specificity (True Negative Rate):** 95%  

You test positive for the disease. What is the probability that you actually have the disease?  

---

### Solution:  

#### Step 1: Identify Given Probabilities  
- Prevalence (Prior Probability of Disease), $P(D)$:  
  $
  P(D) = \frac{1}{1000} = 0.001
  $  
- Sensitivity (Probability of Positive given Disease), $P(\text{Pos} \mid D)$:  
  $
  P(\text{Pos} \mid D) = 0.99
  $  
- False Positive Rate, $P(\text{Pos} \mid \neg D)$:  
  $
  P(\text{Pos} \mid \neg D) = 0.05
  $  

---

#### Step 2: Applying Bayes' Rule  
$
P(D \mid \text{Pos}) = \frac{P(\text{Pos} \mid D) \cdot P(D)}{P(\text{Pos})}
$  

#### Step 3: Marginal Probability of a Positive Test  
$
P(\text{Pos}) = (0.99 \times 0.001) + (0.05 \times 0.999) = 0.05094
$  

#### Step 4: Posterior Probability  
$
P(D \mid \text{Pos}) = \frac{(0.99 \times 0.001)}{0.05094} \approx 0.0194 \text{ or } 1.94\%
$  

### Final Answer:  
The probability that you actually have the disease after testing positive is about **1.94%**.