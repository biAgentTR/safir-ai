import sys, re
with open(sys.argv[1], 'r', encoding='utf-8') as f:
  content = f.read()
template_match = re.search(r'<template>.*</template>', content, flags=re.DOTALL)
with open(sys.argv[2], 'r', encoding='utf-8') as f:
  new_template = f.read()
content = content[:template_match.start()] + new_template + content[template_match.end():]
with open(sys.argv[1], 'w', encoding='utf-8') as f:
  f.write(content)
