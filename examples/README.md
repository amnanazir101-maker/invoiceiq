# Sample receipts

Three receipts from the test split of [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
(Park et al., 2019), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and saved
here as JPEG. The dataset authors blurred store names and other personal details.

| File | CORD id | What it shows |
|---|---|---|
| `receipt-add-ons.jpg` | test-016 | A drink with free add-ons, tax, a rounding line, cash and change |
| `receipt-faded.jpg` | test-020 | Faded thermal print with a service charge and restaurant tax (PB1) |
| `receipt-two-line-items.jpg` | test-026 | Names and prices on separate lines; one line amount is hidden by the blur, so the items do not add up and the receipt goes to review |

Try them in the web app (**Try the sample receipts**) or from the command line:

```bash
invoiceiq extract examples/receipt-add-ons.jpg
```
