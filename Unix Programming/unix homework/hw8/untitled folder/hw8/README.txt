The server, first of all, accept connection from the client and receive a word f
rom client. Then it just look up dictionary to see whether it has similar or the
 exact same one. The calculate similar is to calculate the letter difference between the word in dictionary and the word the server has received. We just select the ten words that has the lowest value of letter difference from the dictionary pool. The lower the letter difference value is, the more similar they are. Finally, server just send word list to the client if the spelling is wrong.  The server ip address is 127.0.0.1



gaopeng@linux1:~/Desktop/hw8/hw8$ ./server 1231 &
[2] 16922
gaopeng@linux1:~/Desktop/hw8/hw8$ ./client 127.0.0.1 1231