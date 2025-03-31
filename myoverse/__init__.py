from icecream import install, ic
import datetime

install()


# Define a function to generate a prefix with ISO timestamp
def timestamp_prefix():
    timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
    return f"{timestamp} | MyoVerse | "


# Configure IceCream to use the timestamp prefix function
ic.configureOutput(includeContext=True, prefix=timestamp_prefix)
