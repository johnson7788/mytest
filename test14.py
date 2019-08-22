def bubble_sort(collection):
   length = len(collection)
   for i in range(length - 1):
      swapped = False
      for j in range(length - 1 - i):
         if collection[j] > collection[j + 1]:
            swapped = True
            collection[j], collection[j + 1] = collection[j + 1], collection[j]
      if not swapped: break  # Stop iteration if the collection is sorted.


   return collection

print(bubble_sort([0, 5, 3, 2, 2]))


def bubble_sort2(lista, n):
   i = 0
   while (i < n - 1):
      j = n - 1
      while (j > i):
         if (lista[j] < lista[j - 1]):
            lista[j - 1], lista[j] = lista[j], lista[j - 1]
         j -= 1
      i += 1
   return lista
print(bubble_sort2([0, 5, 3, 2, 2],5))