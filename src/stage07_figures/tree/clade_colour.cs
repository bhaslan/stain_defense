public static Colour? Format(object attribute)
{
    if (attribute is string hex && hex.Length == 7 && hex[0] == '#')
    {
        int red = Convert.ToInt32(hex.Substring(1, 2), 16);
        int green = Convert.ToInt32(hex.Substring(3, 2), 16);
        int blue = Convert.ToInt32(hex.Substring(5, 2), 16);
        return Colour.FromRgba(red / 255d, green / 255d, blue / 255d, 1d);
    }

    return null;
}
