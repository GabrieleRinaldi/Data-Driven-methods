README 

All'interno della cartella Data-Driven methods troviamo tre cartelle:

•	Methods dove troviamo i metodi data driven,

•	Experimental dataset dove troviamo i dataset sperimentali,

•	Reconstructed dataset dove troviamo i dataset ricostruiti dagli algoritmi della cartella methods.

All'interno delle cartelle dei dataset ricostruiti troviamo le ricostruzioni divise per algoritmo e successivamente anche per tipo di dataset ricostruito.
Nel caso di V2G abbiamo un'ulteriore divisione di ricostruzione per il train ed il test per DMDc.


Per l'utilizzo dei metodi abbiamo bisogno di installare le seguenti librerie:

• matplolib, numpy, scipy, pydmd, csv, scikit-learn, control*, harold*   (*solo per mrDMDc.py)

I file non devono essere messi nella stessa cartella della libreria pydmd.


Per selezionare un dataset è necessario scrivere il percorso nelle variabili dedicate, rispettivamente nominate:

•	"path_for_load_experimental_train" dove vengono caricati i dati sperimentali nei quali si vuole applicare il metodo Data-Driven per il train   

•	"path_for_load_experimental_test" dove vengono caricati i dati sperimentali per il test del metodo Data-Driven
 
•	"path_for_save_reconstructed_train" dove vengono salvati i dati di train ricostruiti dal metodo Data-Driven.

•	"path_for_save_reconstructed_test" dove vengono salvati i dati di test ricostruiti dal metodo Data-Driven.



