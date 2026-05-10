import requests, json

messages = []

print("SHL Assessment Recommender (type 'quit' to exit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    messages.append({'role': 'user', 'content': user_input})
    
    r = requests.post('http://127.0.0.1:8000/chat', json={'messages': messages})
    data = r.json()
    
    print(f"\nAgent: {data['reply']}")
    
    if data['recommendations']:
        print("\nRecommendations:")
        for rec in data['recommendations']:
            print(f"  - {rec['name']} ({rec['test_type']}) → {rec['url']}")
    
    print()
    messages.append({'role': 'assistant', 'content': data['reply']})
    
    if data['end_of_conversation']:
        print("Conversation complete!")
        break