import matplotlib.pyplot as plt
import numpy as np

requests = np.load('../stats/requests.npy')
responses = np.load('../stats/responses.npy')
matched_responses = np.load('../stats/matched_responses.npy')

sum_requests = [0]
sum_responses = [0]
sum_matched_responses = [0]

for req in requests:
    sum_requests.append(sum_requests[-1] + req)
for req in responses:
    sum_responses.append(sum_responses[-1] + req)
for req in matched_responses:
    sum_matched_responses.append(sum_matched_responses[-1] + req)

plt.plot(sum_requests, label='Requests')
plt.plot(sum_responses, label='Responses')
plt.plot(sum_matched_responses, label='Matched Queries')

plt.title('Distribution of requests, responses and matched queries over time')
plt.legend()
plt.savefig('../stats/plot.png')
