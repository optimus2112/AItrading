from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import datetime
import alpaca_tading


# Create an instance of the scheduler
scheduler = BlockingScheduler()

# Define the times at which the job should run (adjust according to your needs)
times = [
    '9:30',  # Market open
    '11:00',  # Mid-morning
    '1:00',  # Early afternoon
    '3:30'   # Just before market close
]

# Schedule the jobs
for time in times:
    hour, minute = map(int, time.split(':'))
    trigger = CronTrigger(hour=hour, minute=minute, second=0, timezone='America/New_York')
    scheduler.add_job(alpaca_tading.main, trigger)

# Start the scheduler
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    pass