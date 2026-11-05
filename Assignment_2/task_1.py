import requests
#User names
usernames = ["fayaz-Hussain", "adnankhan0", "octocat"]

for username in usernames:
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)

    # Verify the status code is 200 before processing
    if response.status_code == 200:
        data = response.json()
        print(f"Login: {data['login']}")
        print(f"Public Repos: {data['public_repos']}")
        print("-" * 25)
    else:
        print(f"Failed to fetch {username} (status code: {response.status_code})")
        print("-" * 25)