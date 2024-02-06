import numpy as np


class Module:
    def __init__(self):
        self.layers = None # []  stack from first layer to the last
        self.input = None
        self.output = None

    def forward(self, input):
        pass


    def parameters(self):
        parameters = []
        for l in self.layers:
            param_arr = np.array(l.parameters())
            parameters.append(param_arr)

        return np.array(parameters , dtype= object)


    def backward(self , loss_grad):  # Backpropagation through entire model after getting loss gradient
        gradient = []
        output_grad_layer = loss_grad
        for l in reversed(self.layers):
            output_grad_layer = l.backward(output_grad_layer)

            if l.get_gradients() == None:  # Activation function
                pass

            else:
                print(l.get_gradients()[0].shape)
                print(l.get_gradients()[1].shape)
                grad_arr = np.array(l.get_gradients())
                gradient.append(grad_arr)

        return np.array(reversed(gradient), dtype = object)





"""
Finally found the one who could help her, she walked straight to the last remaining male student in the classroom.
"Peter, Can you help me solve this physics problem?"  
A female student asked him while pointing out the 20th problem in the exercise sheet. The young man, who is currently writing a Python script, then shifted his focus to the girl.  He gazes at the picture of the inclined and a pulley for a moment before starting to say words carefully. 
"I think you should start drawing a free-body diagram. Then looking for keywords..." He then underlines the phrase "calculate the acceleration".
"which can be used to define the first equation to start solving the problem. In this case, I choose ∑F = ma, however, you still need to find the value for many variables in order to use Newton's second law..." Preaw seemed to understand the problem better while Peter write the variable she needs to find. 
"Thanks, Pete. I'll come to you if there's a problem again" The boy nodded and backed to focus on his work. 

Seeing Praew successfully finish her homework, Pream came to him for help. 
"Ah! So it means not moving at all. I thought acceleration is involved here." Pream felt excited as she found her mistake. "Thanks for your help, Pete. I know you're good at this."  

As Pream walked away, Peter opened his research documents on Particle Physics.  For the last three months, he has been working on his research on how to implement AI systems to detect the sign of each elementary particle. In particular, he is one of the five people in his school to receive a research scholarship. While he was investigating why his model's accuracy isn't meeting with the baseline, suddenly, he heard the announcement from the school's speaker.
"Announced, all territorial defense students gathered at the basketball field at this moment. Announced, all territorial defense students gathered at the basketball field at this moment. " 

The young man lost his attention on the work and walked to the school's corridor lonely. He stood in front of the building's window. 

Looking below from the window of his public high school building, he could see the pole of the Thai flag standing prominently above the group of students gathered in the school courtyard. Those students were his age, and they were wearing khaki-green uniforms and berets.  The chaos occurred due to a large gathering while the teacher, or what he called "commander" was speaking to hurry up the young one.  Thus, they quickly line up before being disciplined.  He then lowered his head and signed for a moment. 
"As you saw from the changing school schedule, every Monday, you will be deployed to the training center..." The commander speaks up with a microphone and a speaker which could be heard from the entire school. 

What he saw below triggered his nostalgia. The scene of his childhood slowly fades in. 

In his mind, he saw a young boy building a robot without the aid of the teacher. He is innocent, pure, and naive. The boy then helped his peers with basic electronics.  It’s a very fun time. 

The new scene then fades in. As time passed, the boy became an adolescent.  This time Peter saw the young man working on his team’s robot.  He’s the captain of the team.  It was the moment before his team won the national competition. At that triumphant moment, he could see the young man, with his peers, putting an arm around each other shoulders with great excitement.  It was one of his greatest satisfaction times.

When the scene slowly fades out for the last time, Peter thought:
"What they saw us when we were boys is an innocent child who should be encouraged to be curious with kindness, but when we grow up into young men, they treat us harshly like we have no heart. Innocence is replaced with responsibility and curiosity with obedience. In other words, society dehumanizes us as we grow up from young innocent boys to discipline young men."

The young student knows his fate and unconsciously thought

"As a part of Generation Z, I once had a beautiful childhood, but within four years, I will be treated like a non-person, punished like a criminal, and embarrassed of being born male even though being male is the first birthday gift I am proud of. "

Distressed by the sound of the commander's announcement about what the boys needed to prepare for the training day, he then walked back to his classroom desolately with an indifferent face. However, in his heart, there is a mix of a sense of injustice and uneasiness.

This is a scene from the novel. I'd like you to create a list of 30 English and language art questions related to this scene.

"""













"""
for x_batch , y_batch in dataset:
    avg_grad = []
    for x , y in x_batch , y_batch:
        
        output = x
        for l in layers:
            output = l.forward(output)
        
        grad = loss_prime(output , y)
        for l in reversed(layers):
            grad = l.backward(grad)
        
        avg_grad.append()
        
    avg_grad = sum(avg_grad) / len(x_batch)
     
    for l in layers:
        grad_l = np.array(l.get_gradients)
        param_l = l.parameters()
        param_l += optimizer(grad_l)  # Not yet!
"""























