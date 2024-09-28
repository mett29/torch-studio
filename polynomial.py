"""
Example adapted from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
"""

import math
import torch


class SimplePolynomial:

    def __init__(self):
        self.dtype = torch.float
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(self.device)

        # Create random input and output data
        self.x = torch.linspace(-math.pi, math.pi, 2000, dtype=self.dtype)
        self.y = torch.sin(self.x)

        # Randomly initialize weights
        self.a = torch.randn((), dtype=self.dtype)
        self.b = torch.randn((), dtype=self.dtype)
        self.c = torch.randn((), dtype=self.dtype)
        self.d = torch.randn((), dtype=self.dtype)

        self.learning_rate = 1e-6

    def manual_backward(self):
        for t in range(2000):
            # Forward pass: compute predicted y
            y_pred = self.a + self.b * self.x + self.c * self.x ** 2 + self.d * self.x ** 3

            # Compute and print loss
            loss = (y_pred - self.y).pow(2).sum().item()
            if t % 100 == 99:
                print(f"step: {t}, loss: {loss}")

            # Backprop to compute gradients of a, b, c, d with respect to loss
            grad_y_pred = 2.0 * (y_pred - self.y)
            grad_a = grad_y_pred.sum()
            grad_b = (grad_y_pred * self.x).sum()
            grad_c = (grad_y_pred * self.x ** 2).sum()
            grad_d = (grad_y_pred * self.x ** 3).sum()

            # Update weights using gradient descent
            self.a -= self.learning_rate * grad_a
            self.b -= self.learning_rate * grad_b
            self.c -= self.learning_rate * grad_c
            self.d -= self.learning_rate * grad_d

        print(f'Result: y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3')

    def automatic_backward(self):
        for t in range(2000):

            self.a.requires_grad = True
            self.b.requires_grad = True
            self.c.requires_grad = True
            self.d.requires_grad = True

            # Forward pass: compute predicted y using operations on Tensors.
            y_pred = self.a + self.b * self.x + self.c * self.x ** 2 + self.d * self.x ** 3

            # Compute and print loss using operations on Tensors.
            # Now loss is a Tensor of shape (1,)
            # loss.item() gets the scalar value held in the loss.
            loss = (y_pred - self.y).pow(2).sum()
            if t % 100 == 99:
                print(f"step: {t}, loss: {loss.item()}")

            # Use autograd to compute the backward pass. This call will compute the
            # gradient of loss with respect to all Tensors with requires_grad=True.
            # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
            # the gradient of the loss with respect to a, b, c, d respectively.
            loss.backward()

            # Manually update weights using gradient descent. Wrap in torch.no_grad()
            # because weights have requires_grad=True, but we don't need to track this
            # in autograd.
            with torch.no_grad():
                self.a -= self.learning_rate * self.a.grad
                self.b -= self.learning_rate * self.b.grad
                self.c -= self.learning_rate * self.c.grad
                self.d -= self.learning_rate * self.d.grad

                # Manually zero the gradients after updating weights
                self.a.grad = None
                self.b.grad = None
                self.c.grad = None
                self.d.grad = None

        print(f'Result: y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3')


if __name__ == "__main__":

    simple_polynomial = SimplePolynomial()
    simple_polynomial.manual_backward()
    simple_polynomial.automatic_backward()
