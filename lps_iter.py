def longestPalindrome(s):
	if(len(s) == 0):
		return null
	if(len(s) == 1):
		return s
	longest = s[0:1]
	for i in range(len(s)):
		tmp = helper(s,i,i)
		if(len(tmp)>len(longest)):
			longest = tmp

		tmp = helper(s,i,i+1)
		if(len(tmp)>len(longest)):
			longest = tmp
	return longest

def helper(s,b,e):
	while(b>=0 and e<=len(s)-1 and s[b] == s[e]):
		b -= 1
		e += 1
	return s[b+1:e]

def main():
	s = 'bbbbbaaaaaaabbb'
	stri = longestPalindrome(s)
	print(stri)
	print(len(stri))

if __name__== "__main__":
  main()


