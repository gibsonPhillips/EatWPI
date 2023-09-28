export interface FoodItems {
    readonly value: string,
    readonly label: string
}

export const foodItems: readonly FoodItems[] = [
    {value: "bread", label: "Bread"},
    {value: "cheese", label: "Cheese"},
]