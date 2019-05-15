import torch
import random


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def prepare_data(data, device):
    imgs, captions, captions_lens, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for img in imgs:
        img = img[sorted_cap_indices].to(device)
        real_imgs.append(img)

    captions = captions[sorted_cap_indices].squeeze().to(device)
    sorted_cap_lens = sorted_cap_lens.to(device)
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]


# right now this method is available for only COCO dataset
def r_precision(dataset, G, z_dim, cnn_model, rnn_model, device,
                num_images=30000):
    G.eval()
    cnn_model.eval()
    rnn_model.eval()

    noise = torch.FloatTensor(1, z_dim).to(device)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=100, drop_last=True, shuffle=True, num_workers=4)
    data_iter = iter(dataloader)
    score = 0

    with torch.no_grad():
        for i in range(num_images):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data = next(data_iter)

            _, captions, cap_lens, \
                class_ids, keys = prepare_data(data, device)
            ind = random.randrange(100)
            # calculate 100 sent_emb
            words_embs, sent_emb = rnn_model(captions, cap_lens)
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            noise.data.normal_()
            target_sent_emb = sent_emb.detach()[ind].unsqueeze(0)
            target_words_embs = words_embs.detach()[ind].unsqueeze(0)
            target_mask = mask.detach()[ind].unsqueeze(0)
            fake_img = \
                G(noise, target_sent_emb, target_words_embs, target_mask)[0]
            _, sent_code = cnn_model(fake_img[-1])
            sent_code = sent_code.repeat(100, 1)

            sim = cosine_similarity(sent_code, sent_emb)
            max_sim_ind = sim.argmax().item()

            if max_sim_ind == ind:
                score += 1

    return score / num_images
