def test(model, test_loader):
    model.to(device)
    model.eval()

    test_losses = []
    test_loss = 0

    with torch.no_grad():
        for batch_id, (img, carbon, _, _) in enumerate(test_loader):
            img, carbon = img.to(torch.float32), carbon.to(torch.float32)
            img, carbon = img.to(device), carbon.to(device)
            output = model(img)
            output=torch.squeeze(output)
            test_losses.append(loss_fn(output, carbon).item())


    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(sum(test_losses)/len(test_losses)))

