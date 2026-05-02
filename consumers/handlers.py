from agent.utils.db import get_pg_conn, release_pg_conn

from agent.utils.db import get_pg_conn, release_pg_conn

def handle_product_sync(data: dict):
    conn = None
    cur = None
    
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        
        source_product_id = data.get('productId')
        name = data.get('name')
        rating = data.get('rating')
        rated_quantity = data.get('ratedQuantity')
        brand = data.get('brand')
        category = data.get('category')
        
        update_type = data.get('updateType')
        update_entity = data.get('updateEntity')
        variants = data.get('variantDetails', [])
        
        if not source_product_id or not update_type or not update_entity:
            print("Warning: Missing productId, updateType, or updateEntity in data, skipping.")
            return

        print(f"Processing {update_type} ({update_entity}) event for product Source ID: {source_product_id}")

        # ==========================================
        # 1. CREATE 
        # ==========================================
        if update_type == "CREATE":
            
            if update_entity == "PRODUCT":

                cur.execute("SELECT id FROM product WHERE ps_product_id = %s;", (source_product_id,))
                if cur.fetchone():
                    print(f"Error: Product {source_product_id} already exists. Cannot CREATE.")
                    return 
                
                cur.execute("""
                    INSERT INTO product (name, ps_product_id, rating, rated_quantity, brand, category) 
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
                """, (name, source_product_id, rating, rated_quantity, brand, category))
                internal_product_id = cur.fetchone()[0]
                
                for var in variants:
                    source_variant_id = var.get('variantId')
                    color = var.get('color')
                    price = var.get('price')
                    image = var.get('mainImage')
                    
                    cur.execute("""
                        INSERT INTO variant (product_id, color, price, image, ps_product_id, ps_variant_id)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """, (internal_product_id, color, price, image, source_product_id, source_variant_id))

            elif update_entity == "VARIANT":
                cur.execute("SELECT id FROM product WHERE ps_product_id = %s;", (source_product_id,))
                prod_row = cur.fetchone()
                
                if not prod_row:
                    print(f"Error: Parent Product {source_product_id} not found in DB. Cannot CREATE variant.")
                    return
                
                internal_product_id = prod_row[0]
                
                for var in variants:
                    source_variant_id = var.get('variantId')
                    
                    cur.execute("SELECT id FROM variant WHERE ps_variant_id = %s;", (source_variant_id,))
                    if cur.fetchone():
                        print(f"Warning: Variant {source_variant_id} already exists. Skipping CREATE for this variant.")
                        continue
                        
                    color = var.get('color')
                    price = var.get('price')
                    image = var.get('mainImage')
                    
                    cur.execute("""
                        INSERT INTO variant (product_id, color, price, image, ps_product_id, ps_variant_id)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """, (internal_product_id, color, price, image, source_product_id, source_variant_id))

        # ==========================================
        # 2. UPDATE
        # ==========================================
        elif update_type == "UPDATE":
            
            if update_entity == "PRODUCT":
                
                cur.execute("SELECT id FROM product WHERE ps_product_id = %s;", (source_product_id,))
                if not cur.fetchone():
                    print(f"Error: Product {source_product_id} not found. Cannot UPDATE.")
                    return
                
                cur.execute("""
                    UPDATE product 
                    SET name = COALESCE(%s, name),
                        rating = COALESCE(%s, rating),
                        rated_quantity = COALESCE(%s, rated_quantity),
                        brand = COALESCE(%s, brand),
                        category = COALESCE(%s, category)
                    WHERE ps_product_id = %s;
                """, (name, rating, rated_quantity, brand, category, source_product_id))

            elif update_entity == "VARIANT":
                for var in variants:
                    source_variant_id = var.get('variantId')
                    
                    cur.execute("SELECT id FROM variant WHERE ps_variant_id = %s;", (source_variant_id,))
                    if not cur.fetchone():
                        print(f"Error: Variant {source_variant_id} not found. Cannot UPDATE.")
                        continue
                    
                    color = var.get('color')
                    price = var.get('price')
                    image = var.get('mainImage') 
                    
                    cur.execute("""
                        UPDATE variant 
                        SET color = COALESCE(%s, color),
                            price = COALESCE(%s, price),
                            image = COALESCE(%s, image)
                        WHERE ps_variant_id = %s;
                    """, (color, price, image, source_variant_id))

        # ==========================================
        # 3. DELETE
        # ==========================================
        elif update_type == "DELETE":
            if update_entity == "PRODUCT":
                cur.execute("DELETE FROM variant WHERE ps_product_id = %s;", (source_product_id,))
                cur.execute("DELETE FROM product WHERE ps_product_id = %s;", (source_product_id,))
                print(f"Deleted entire product Source ID: {source_product_id}")
                
            elif update_entity == "VARIANT":
                if not variants:
                    print(f"Warning: Received DELETE VARIANT event for Source ID: {source_product_id} but variant list is empty.")
                else:
                    for var in variants:
                        source_variant_id = var.get('variantId')
                        cur.execute("DELETE FROM variant WHERE ps_variant_id = %s;", (source_variant_id,))
                    
                    cur.execute("SELECT COUNT(*) FROM variant WHERE ps_product_id = %s;", (source_product_id,))
                    remaining_variants = cur.fetchone()[0]
                    
                    if remaining_variants == 0:
                        cur.execute("DELETE FROM product WHERE ps_product_id = %s;", (source_product_id,))
                        print(f"Deleted entire product Source ID: {source_product_id} (0 variants remaining)")
                    else:
                        print(f"Deleted specific variants for product Source ID: {source_product_id}. Remaining variants: {remaining_variants}")
            else:
                print(f"Warning: Unknown UpdateEntity '{update_entity}' for DELETE operation.")

        else:
            print(f"Warning: Unknown updateType received: {update_type}")
            return

        conn.commit()
        print(f"Successfully processed {update_type} ({update_entity}) event for Source ID: {source_product_id}")

    except Exception as e:
        print(f"Error executing database operations: {e}")
        if conn:
            conn.rollback()
            
    finally:
        if cur:
            cur.close()
        if conn:
            release_pg_conn(conn)

def handle_inventory_sync(data: dict):
    """
    Handles the UpdateCartProductDetailEvent to sync inventory stock.
    """
    conn = None
    cur = None
    
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        
        product_details = data.get('productDetails', [])
        
        if not product_details:
            print("Warning: Received inventory sync event but productDetails is empty.")
            return

        print(f"Processing inventory sync for {len(product_details)} variants.")

        for detail in product_details:
            source_variant_id = detail.get('variantId')
            in_stock = detail.get('inStock')
            
            if source_variant_id and in_stock is not None:
                cur.execute("""
                    UPDATE variant 
                    SET stock = %s 
                    WHERE ps_variant_id = %s;
                """, (in_stock, source_variant_id))
                
        conn.commit()
        print("Successfully synchronized inventory stock.")

    except Exception as e:
        print(f"Error executing database operations for inventory sync: {e}")
        if conn:
            conn.rollback()
            
    finally:
        if cur:
            cur.close()
        if conn:
            release_pg_conn(conn)

def handle_rating_sync(data: dict):
    """
    Handles rating updates from Kafka and synchronizes the average rating 
    to the AI product table.
    """
    conn = None
    cur = None
    
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        
        source_product_id = data.get('productId')
    
        new_rating = data.get('averageRating') 
        new_rated_quantity = data.get('ratedQuantity')
        
        if not source_product_id or new_rating is None:
            print("Warning: Missing productId or averageRating in data, skipping.")
            return

        print(f"Processing rating sync for Product Source ID: {source_product_id}")

        cur.execute("""
            UPDATE product 
            SET rating = %s,
                rated_quantity = %s
            WHERE ps_product_id = %s;
        """, (new_rating, new_rated_quantity, source_product_id))

        conn.commit()
        print(f"Successfully synced new rating ({new_rating}) for Source ID: {source_product_id}")

    except Exception as e:
        print(f"Error executing database operations for rating sync: {e}")
        if conn:
            conn.rollback()
            
    finally:
        if cur:
            cur.close()
        if conn:
            release_pg_conn(conn)