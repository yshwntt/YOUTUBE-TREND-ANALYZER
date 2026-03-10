#1. Get all customers whose first name starts with 'J' and who are active.

SELECT distinct first_name From sakila.actor
where first_name like 'j%';

#2. Find all films where the title contains the word 'ACTION' or the description contains 'WAR'.

SELECT title, description
from sakila.film
where title LIKE'%ACTION%' 
or  description like '%WAR%';


#3. List all customers whose last name is not 'SMITH' and whose first name ends with 'a'.

select first_name, last_name
from sakila.actor
where last_name <> 'SMITH'
AND first_name LIKE '%a';

#4. Get all films where the rental rate is greater than 3.0 and the replacement cost is not null.

select title, rental_rate, replacement_cost
from sakila.film
where rental_rate >= 3.0
AND replacement_cost IS NOT NULL;

#5. Count how many customers exist in each store who have active status = 1.

select distinct store_id, count(customer_id) as active_customers
from sakila.customer
where active = 1
group by store_id;

#6. Show distinct film ratings available in the film table.

select distinct rating
from sakila.film; 

#7. Find the number of films for each rental duration where the average length is more than 100 minutes.

select rental_duration, count(title) as film_count
from sakila.film
group by rental_duration
having AVG(length) >= 100;

#8. List payment dates and total amount paid per date, but only include days where more than 100 payments were made.

select date(payment_date) as payment_date, sum(amount) as total_amount, count(*) as payment
from sakila.payment
group by date(payment_date)
having count(*) > 100;   

#9. Find customers whose email address is null or ends with '.org'.
select email
from sakila.customer
where email is null 
OR email LIKE '%.org';

#10. List all films with rating 'PG' or 'G', and order them by rental rate in descending order.

select rating, title, rental_rate as ren
from sakila.film
where rating = 'PG' or rating = 'G'
order by rental_rate desc;


#11. Count how many films exist for each length where the film title starts with 'T' and the count is more than 5.

/*select length, count(title) as filim_count
from sakila.film
where title like 'T%'
group by length
having count(title) > 5;*/

#12. List all actors who have appeared in more than 10 films.

#13. Find the top 5 films with the highest rental rates and longest lengths combined, ordering by rental rate first and length second.

select title, rental_rate, length
from sakila.film
order by rental_rate desc, length desc
limit 5;
 
#14. Show all customers along with the total number of rentals they have made, ordered from most to least rentals.

#15. List the film titles that have never been rented.

select title 
from sakila.film


