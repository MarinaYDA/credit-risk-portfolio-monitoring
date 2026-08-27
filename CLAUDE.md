## Professional Code Standards
Apply these checks automatically before treating any day's task as done -
don't wait for me to catch it:
- Portability: never hardcode a path tied to my username or machine (e.g.
  C:\Users\yersh\...). Use Path(__file__).resolve()-based paths or paths
  relative to the repo root, so a fresh git clone runs unmodified on
  anyone else's machine - including a recruiter's or hiring manager's.
- Reproducibility: if a step needs manual setup beyond
  pip install -r requirements.txt, write it in README.md, not just tell
  me in chat.
- Validate assumptions instead of just stating them: if code depends on
  an assumption (a date format, a column being non-null, a value range),
  check it programmatically and report when it doesn't hold, instead of
  assuming it's fine.
- No secrets, tokens, or personal file paths in committed code, ever.
- Before calling a day "done," self-check: would this run correctly right
  after a stranger clones this repo on their own machine? If not, fix it
  before marking the day complete, don't wait for me to notice.
