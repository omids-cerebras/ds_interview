# Probability Questions

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