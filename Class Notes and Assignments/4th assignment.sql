#1. List all customers along with the films they have rented.

select c.customer_id, f.title
from sakila.customer c
inner join sakila.rental r on c.customer_id = r.customer_id
inner join sakila.inventory i on r.inventory_id = i.inventory_id
inner join sakila.film f on i.film_id = f.film_id;


#2. List all customers and show their rental count, including those who haven't rented any films.

select c.customer_id, count(r.rental_id) as rental_count
from sakila.customer c
left join sakila.rental r on c.customer_id = r.customer_id
group by c.customer_id;

#3. Show all films along with their category. Include films that don't have a category assigned.

select f.title, c.name
from sakila.film f
left join sakila.film_category fc on f.film_id = fc.film_id
left join sakila.category c on fc.category_id = c.category_id;


#4. Show all customers and staff emails from both customer and staff tables using a full outer join (simulate using LEFT + RIGHT + UNION).

select c.email
from sakila.customer c
left join sakila.staff s on 1 = 0
union
select s.email
from sakila.staff s
left join sakila.customer c on 1 = 0;

#5. Find all actors who acted in the film "ACADEMY DINOSAUR".

select a.first_name
from sakila.actor a
inner join sakila.film_actor fa on a.actor_id = fa.actor_id
inner join sakila.film f on fa.film_id = f.film_id
where f.title = 'academy dinosaur';


#6. List all stores and the total number of staff members working in each store, even if a store has no staff.

select st.store_id, count(sf.staff_id) as staff_count
from sakila.store st
left join sakila.staff sf on st.store_id = sf.store_id
group by st.store_id;

#7. List the customers who have rented films more than 5 times. Include their name and total rental count.

select c.customer_id, count(r.rental_id) as rental_count
from sakila.customer c
inner join sakila.rental r on c.customer_id = r.customer_id
group by c.customer_id
having count(r.rental_id) > 5;

