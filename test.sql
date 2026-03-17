select user_id
from (
    select user_id,
           date,
           date - row_number() over(partition by user_id order by date) as flag
    from login
) t
group by user_id, flag
having count(*) >= 7;