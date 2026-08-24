"""
gen_password_hash.py
---------------------
Run this LOCALLY to turn a password you choose into a secure hash, without
that plaintext password ever being written to a file, committed to git, or
sent anywhere. You then put the resulting hash (not the password) into the
ADMIN_PASSWORD_HASH environment variable that web_stock_alerts.py reads.

USAGE
-----
    python gen_password_hash.py

It will prompt you to type a password (hidden, not echoed to the screen),
then print a hash that looks like:

    scrypt:32768:8:1$abcXYZ...$9f8e7d6c5b4a...

Copy that ENTIRE string and set it as an environment variable called
ADMIN_PASSWORD_HASH (see README.md for exactly how to do this locally and
on your hosting platform).

IMPORTANT: pick a password you have NEVER used anywhere else - not your
email password, not a password reused from another site. This app will be
reachable from the internet once deployed, so it deserves its own unique,
strong password.
"""

import getpass

from werkzeug.security import generate_password_hash


def main():
    print("This generates a password HASH for logging into your Stock Alert")
    print("Dashboard. The plaintext password you type is never saved or shown")
    print("again - only the hash below gets stored.\n")

    pw1 = getpass.getpass("Choose a new password (use a NEW one, not reused): ")
    pw2 = getpass.getpass("Type it again to confirm: ")

    if pw1 != pw2:
        print("\nPasswords didn't match - run this again.")
        return
    if len(pw1) < 8:
        print("\nPlease choose a password of at least 8 characters - run this again.")
        return

    hashed = generate_password_hash(pw1)
    print("\nSet this as your ADMIN_PASSWORD_HASH environment variable:\n")
    print(hashed)
    print("\n(See README.md for how to set environment variables locally and on your hosting platform.)")


if __name__ == "__main__":
    main()
