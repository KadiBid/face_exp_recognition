docker run -it --rm -d -p 80:80 --name web -v $(pwd)/app:/usr/share/nginx/html nginx
