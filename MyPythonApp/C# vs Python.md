C# vs Python

1) Değişkenler & Veri Tipleri
| C#                         | Python             | Not                                           |
| -------------------------- | ------------------ | --------------------------------------------- |
| `int x = 5;`               | `x = 5`            | Python’da tip belirtmeye gerek yok (dinamik). |
| `double pi = 3.14;`        | `pi = 3.14`        | Float otomatik.                               |
| `string name = "Anatoly";` | `name = "Anatoly"` | Hem `' '` hem `" "` kullanılabilir.           |
| `bool isActive = true;`    | `is_active = True` | Büyük harfle `True/False`.                    |
| `char c = 'A';`            | `c = 'A'`          | Tek karakter diye ayrı tip yok, hepsi `str`.  |

2) Listeler (ArrayList / List eşdeğeri)
| C#                                  | Python            | Açıklama                  |
| ----------------------------------- | ----------------- | ------------------------- |
| `List<int> list = new List<int>();` | `list = []`       | Dinamik liste             |
| `list.Add(10);`                     | `list.append(10)` | Ekleme                    |
| `list[0]`                           | `list[0]`         | Index erişimi             |
| `list.Contains(10)`                 | `10 in list`      | İçeriyor mu               |
| `list.Remove(10)`                   | `list.remove(10)` | İlk bulduğu elemanı siler |
| `list.Count`                        | `len(list)`       | Eleman sayısı             |

3) Dictionary (HashMap eşdeğeri)
| C#                                         | Python            | Açıklama                 |
| ------------------------------------------ | ----------------- | ------------------------ |
| `var dict = new Dictionary<string,int>();` | `d = {}`          | Key-Value map            |
| `dict["a"] = 1;`                           | `d["a"] = 1`      | Ekleme                   |
| `dict.ContainsKey("a")`                    | `"a" in d`        | Key kontrol              |
| `dict.ContainsValue(1)`                    | `1 in d.values()` | Value kontrol            |
| `dict["a"]`                                | `d.get("a")`      | Get (yoksa `None` döner) |
| `dict.Remove("a");`                        | `del d["a"]`      | Silme                    |

4) Set (HashSet eşdeğeri)
| C#                              | Python        | Açıklama            |
| ------------------------------- | ------------- | ------------------- |
| `var set = new HashSet<int>();` | `s = set()`   | Benzersiz elemanlar |
| `set.Add(5);`                   | `s.add(5)`    | Ekleme              |
| `set.Contains(5)`               | `5 in s`      | İçeriyor mu         |
| `set.Remove(5)`                 | `s.remove(5)` | Silme               |

5) Döngüler
C#
    for (int i = 0; i < 5; i++) {
        Console.WriteLine(i);
    }
------------------------------
    foreach (var num in list) {
        Console.WriteLine(num);
    }


------------------------------
    int x = 5;
    while (x > 0) {
        x--;
    }

Python
    for i in range(5):
        print(i)
------------------------------
    for num in list:
        print(num)

    for i, fruit in enumerate(fruits, start=1):
    print(i, fruit)
------------------------------
    x = 5
    while x > 0:
        x -= 1

6) Fonksiyonlar
C#
    int Add(int a, int b) {
        return a + b;
    }

Python
    def add(a, b):
        return a + b
Python’da tip belirtmeye gerek yok ama istersen type hint ekleyebilirsin:
    def add(a: int, b: int) -> int:
        return a + b

7) Özel Noktalar (C# vs Python farkları)
* Indentation → Python’da süslü parantez yok, girintiyle bloklar belirleniyor.

if x > 5:
    print("Büyük")
else:
    print("Küçük")


* Null karşılığı → C# null, Python None.

* Boolean → C# true/false, Python True/False.

* String interpolation
    C#: $"Hello {name}"
    Python: f"Hello {name}"

* Print → C#: Console.WriteLine(), Python: print().