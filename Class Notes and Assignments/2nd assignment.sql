#1. Identify if there are duplicates in Customer table. Don't use customer id to check the duplicates

select first_name, last_name, email, store_id, count(*) as dupcnt
from sakila.customer
group by first_name, last_name, email, store_id
having count(*)>1;


#2. Number of times letter 'a' is repeated in film descriptions

select title, (length(description) - length(replace(description, 'a', ''))) as a_count 
from sakila.film;



#3. Number of times each vowel is repeated in film descriptions 

select title, (length(description) - length(replace(description, 'a', ''))) as a_count, (length(description) - length(replace(description, 'e', ''))) as e_count,
(length(description) - length(replace(description, 'i', ''))) as i_count, (length(description) - length(replace(description, 'o', ''))) as o_count, (length(description) - length(replace(description, 'u', ''))) as u_count    
from sakila.film;

#4. Display the payments made by each customer
 #       1. Month wise

select customer_id, month(payment_date) as month, sum(amount) as tpaymt
from sakila.payment
group by customer_id, month(payment_date);

#      2. Year wise
  
select customer_id, year(payment_date) as year, sum(amount) as tpaymt
from sakila.payment
group by customer_id, year(payment_date);

   #     3. Week wise
   
select customer_id, week(payment_date) as week, sum(amount) as tpaymt
from sakila.payment
group by customer_id, week(payment_date);

#5. Check if any given year is a leap year or not. You need not consider any table from sakila database. Write within the select query with hardcoded date

select 2024 as year, 
case
when(2024 % 400 = 0) or (2024 % 4 = 0 and 2024 % 100 <> 0)
then 'leap year'
else 'not a leap year'
end as result;

#6. Display number of days remaining in the current year from today.

select datediff('2026-12-31', '2026-01-28') as days_rem;

#7. Display quarter number(Q1,Q2,Q3,Q4) for the payment dates from payment table. 

select payment_date, quarter(payment_date) as quarter
from sakila.payment;

#8. Display the age in year, months, days based on your date of birth. 
  # For example: 21 years, 4 months, 12 days
  
  select concat(timestampdiff(year, '2000-01-01', curdate()), 'years,', timestampdiff(month, '2000-01-01', curdate()) % 12, 'months,') as age;
