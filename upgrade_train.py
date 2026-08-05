import re

def main():
    with open('/home/yug/My_Practice/spacetech/train.py', 'r') as f:
        content = f.read()

    # Update SingleAgentWrapper instantiation
    content = content.replace('SingleAgentWrapper(num_positions=20, max_steps=100', 'SingleAgentWrapper(num_positions=360, max_steps=360')
    content = content.replace('ep_steps=50', 'ep_steps=360')

    with open('/home/yug/My_Practice/spacetech/train.py', 'w') as f:
        f.write(content)
        
    print("Upgraded train.py")

if __name__ == '__main__':
    main()
